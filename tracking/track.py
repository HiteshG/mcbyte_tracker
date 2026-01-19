"""
Track Management and Matching
=============================
Core track class and association utilities for McByte tracker.
"""

import numpy as np
from enum import IntEnum
from typing import List, Optional, Tuple, Dict
from collections import deque

from .kalman_filter import KalmanFilter


class TrackState(IntEnum):
    """Track lifecycle states."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class Track:
    """
    Single object track with Kalman filtering.
    
    Maintains state history and provides coordinate conversions.
    """
    
    # Shared Kalman filter instance
    _kalman_filter = KalmanFilter()
    _id_counter = 0
    
    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        class_id: int = 0,
        buffer_size: int = 30
    ):
        """
        Initialize a new track.
        
        Args:
            tlwh: [top, left, width, height] bounding box
            score: Detection confidence score
            class_id: Object class ID
            buffer_size: History buffer length
        """
        # Bounding box
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        
        # Kalman filter state
        self.mean, self.covariance = self._kalman_filter.initiate(
            self._tlwh_to_xywh(tlwh)
        )
        
        # Track metadata
        self.is_activated = False
        self.score = score
        self.class_id = class_id
        
        # Lifecycle tracking
        self.state = TrackState.NEW
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Tracking history
        self.tracklet_len = 0
        self.history = deque(maxlen=buffer_size)
    
    @staticmethod
    def _tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert [t,l,w,h] to [cx,cy,w,h]."""
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    @staticmethod
    def _xywh_to_tlwh(xywh: np.ndarray) -> np.ndarray:
        """Convert [cx,cy,w,h] to [t,l,w,h]."""
        ret = xywh.copy()
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlwh(self) -> np.ndarray:
        """Get [top, left, width, height] from Kalman state."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get [top, left, bottom, right]."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def xywh(self) -> np.ndarray:
        """Get [center_x, center_y, width, height]."""
        return self.mean[:4].copy() if self.mean is not None else self._tlwh_to_xywh(self._tlwh)
    
    @property
    def end_frame(self) -> int:
        """Get last seen frame."""
        return self.frame_id
    
    def predict(self) -> None:
        """Propagate state using Kalman filter."""
        self.mean, self.covariance = self._kalman_filter.predict(
            self.mean, self.covariance
        )
    
    def update(self, detection, frame_id: int) -> None:
        """
        Update track with new detection.
        
        Args:
            detection: Detection object or (tlwh, score) tuple
            frame_id: Current frame number
        """
        if hasattr(detection, 'tlwh'):
            tlwh = detection.tlwh
            score = detection.confidence
        else:
            tlwh, score = detection
        
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        
        # Kalman update
        self.mean, self.covariance = self._kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xywh(tlwh)
        )
        
        # Update state
        self.tracklet_len += 1
        self.frame_id = frame_id
        self.state = TrackState.TRACKED
        self.is_activated = True
        
        # Record history
        self.history.append(self.tlwh.copy())
    
    def activate(self, frame_id: int) -> None:
        """Start a new track."""
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.history.append(self._tlwh.copy())
    
    def reactivate(self, detection, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a lost track."""
        if hasattr(detection, 'tlwh'):
            tlwh = detection.tlwh
            score = detection.confidence
        else:
            tlwh, score = detection
        
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.mean, self.covariance = self._kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xywh(tlwh)
        )
        
        self.score = score
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.state = TrackState.TRACKED
        self.is_activated = True
        
        if new_id:
            Track._id_counter += 1
            self.track_id = Track._id_counter
        
        self.history.append(self.tlwh.copy())
    
    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = TrackState.LOST
    
    def mark_removed(self) -> None:
        """Mark track as removed."""
        self.state = TrackState.REMOVED
    
    @staticmethod
    def reset_id() -> None:
        """Reset the track ID counter."""
        Track._id_counter = 0
    
    @staticmethod
    def multi_predict(tracks: List['Track']) -> None:
        """Predict for multiple tracks."""
        for track in tracks:
            track.predict()
    
    def __repr__(self) -> str:
        return f"Track_{self.track_id}(state={self.state.name}, frames={self.start_frame}-{self.end_frame})"


# =============================================================================
# Matching Utilities
# =============================================================================

def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes_a: [N, 4] in tlbr format
        boxes_b: [M, 4] in tlbr format
        
    Returns:
        [N, M] IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    # Compute intersections
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    
    wh = np.maximum(br - tl, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    # Compute areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    
    # Compute IoU
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


def iou_distance(
    tracks: List[Track],
    detections: List,
    use_prediction: bool = True
) -> np.ndarray:
    """
    Compute IoU distance matrix between tracks and detections.
    
    Args:
        tracks: List of Track objects
        detections: List of Detection objects or (tlwh, score) tuples
        use_prediction: Use predicted position instead of last detection
        
    Returns:
        [N_tracks, N_detections] cost matrix (1 - IoU)
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))
    
    # Extract boxes
    if use_prediction:
        track_boxes = np.array([t.tlbr for t in tracks])
    else:
        track_boxes = np.array([t.tlbr for t in tracks])
    
    if hasattr(detections[0], 'tlbr'):
        det_boxes = np.array([d.tlbr for d in detections])
    else:
        det_boxes = np.array([
            np.array([d[0][0], d[0][1], d[0][0] + d[0][2], d[0][1] + d[0][3]])
            for d in detections
        ])
    
    iou = iou_batch(track_boxes, det_boxes)
    return 1 - iou


def fuse_score(
    cost_matrix: np.ndarray,
    detections: List,
    coef: float = 1.0
) -> np.ndarray:
    """
    Fuse detection confidence into cost matrix.
    
    Lower confidence = higher cost.
    
    Args:
        cost_matrix: [N, M] cost matrix
        detections: List of detections
        coef: Weighting coefficient
        
    Returns:
        Modified cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    if hasattr(detections[0], 'confidence'):
        scores = np.array([d.confidence for d in detections])
    else:
        scores = np.array([d[1] for d in detections])
    
    iou_sim = 1 - cost_matrix
    fused = iou_sim * np.exp(coef * (scores - 1))
    
    return 1 - fused


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float = 0.8
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve linear assignment problem (Hungarian algorithm).
    
    Args:
        cost_matrix: [N, M] cost matrix
        thresh: Maximum cost threshold
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
    """
    from scipy.optimize import linear_sum_assignment
    
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_dets = list(range(cost_matrix.shape[1]))
    
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= thresh:
            matches.append((row, col))
            unmatched_tracks.remove(row)
            unmatched_dets.remove(col)
    
    return matches, unmatched_tracks, unmatched_dets
