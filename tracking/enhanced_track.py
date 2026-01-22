"""
Enhanced Track Class for Identity-Preserving Tracking
======================================================
Combines:
- Unscented Kalman Filter for state estimation
- Appearance features for re-identification
- Mask-based features for occlusion handling
- Track memory for lost track recovery
"""

import numpy as np
from enum import IntEnum
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import deque

from .ukf import UnscentedKalmanFilter, AdaptiveUKF
from .appearance import AppearanceFeature


class TrackState(IntEnum):
    """Track lifecycle states."""
    NEW = 0        # Just created, not yet confirmed
    TRACKED = 1    # Actively tracked
    LOST = 2       # Temporarily lost, searching
    REMOVED = 3    # Permanently removed


@dataclass
class TrackMemory:
    """
    Memory of a track for re-identification.
    Stores information needed to recover lost tracks.
    """
    track_id: int
    class_id: int
    last_bbox: np.ndarray
    last_position: np.ndarray  # [x, y]
    last_velocity: np.ndarray  # [vx, vy]
    appearance_features: List[AppearanceFeature] = field(default_factory=list)
    mask_features: List[np.ndarray] = field(default_factory=list)
    lost_frame: int = 0
    confidence: float = 1.0
    
    def predict_position(self, frames_elapsed: int) -> np.ndarray:
        """Predict where track would be after N frames."""
        return self.last_position + self.last_velocity * frames_elapsed


class EnhancedTrack:
    """
    Enhanced track with UKF and appearance features.
    
    Features:
    - Non-linear state estimation via UKF
    - Appearance feature gallery for ReID
    - Mask-based identity verification
    - Occlusion detection and handling
    """
    
    _count = 0  # Global track ID counter
    
    # Shared UKF instance for efficiency
    _ukf = None
    
    @classmethod
    def reset_id_counter(cls):
        """Reset track ID counter."""
        cls._count = 0
    
    @classmethod
    def get_ukf(cls) -> UnscentedKalmanFilter:
        """Get shared UKF instance."""
        if cls._ukf is None:
            cls._ukf = AdaptiveUKF()
        return cls._ukf
    
    def __init__(
        self,
        detection: Tuple[np.ndarray, float, int],
        frame_id: int,
        appearance_feature: Optional[AppearanceFeature] = None
    ):
        """
        Initialize track from detection.
        
        Args:
            detection: (tlwh, confidence, class_id)
            frame_id: Current frame number
            appearance_feature: Initial appearance feature
        """
        tlwh, confidence, class_id = detection
        
        # Assign unique ID
        EnhancedTrack._count += 1
        self.track_id = EnhancedTrack._count
        
        # Class information
        self.class_id = class_id
        self.score = confidence
        
        # Initialize UKF state
        xywh = self._tlwh_to_xywh(tlwh)
        ukf = self.get_ukf()
        self.mean, self.covariance = ukf.initiate(xywh)
        
        # Track state
        self.state = TrackState.NEW
        self.is_activated = False
        
        # Timing
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.time_since_update = 0
        
        # Appearance features
        self.appearance_history: deque = deque(maxlen=30)
        if appearance_feature is not None:
            self.appearance_history.append(appearance_feature)
        self.smooth_appearance: Optional[AppearanceFeature] = appearance_feature
        
        # Mask features
        self.mask_history: deque = deque(maxlen=10)
        self.current_mask: Optional[np.ndarray] = None
        self.mask_id: Optional[int] = None
        
        # Occlusion handling
        self.occlusion_count = 0
        self.is_occluded = False
        
        # Position history for trajectory
        self.position_history: deque = deque(maxlen=50)
        self._update_position_history()
        
        # Track quality metrics
        self.hit_streak = 0
        self.age = 0
    
    @staticmethod
    def _tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, w, h] to [cx, cy, w, h]."""
        ret = tlwh.copy()
        ret[:2] = tlwh[:2] + tlwh[2:] / 2
        return ret
    
    @staticmethod
    def _xywh_to_tlwh(xywh: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x1, y1, w, h]."""
        ret = xywh.copy()
        ret[:2] = xywh[:2] - xywh[2:] / 2
        return ret
    
    def _update_position_history(self):
        """Update position history with current state."""
        bbox = self.get_ukf().state_to_bbox(self.mean)
        self.position_history.append(bbox[:2].copy())
    
    def predict(self):
        """
        Run UKF prediction step.
        Updates state estimate based on motion model.
        """
        ukf = self.get_ukf()
        self.mean, self.covariance = ukf.predict(self.mean, self.covariance)
        
        self.age += 1
        self.time_since_update += 1
        
        # Update occlusion status
        if self.time_since_update > 1:
            self.occlusion_count += 1
            self.is_occluded = True
        
        self._update_position_history()
    
    def update(
        self,
        detection: Tuple[np.ndarray, float, int],
        frame_id: int,
        appearance_feature: Optional[AppearanceFeature] = None,
        mask: Optional[np.ndarray] = None
    ):
        """
        Update track with new detection.
        
        Args:
            detection: (tlwh, confidence, class_id)
            frame_id: Current frame number
            appearance_feature: Appearance feature from detection
            mask: Segmentation mask
        """
        tlwh, confidence, class_id = detection
        
        # UKF update
        xywh = self._tlwh_to_xywh(tlwh)
        ukf = self.get_ukf()
        self.mean, self.covariance = ukf.update(
            self.mean, self.covariance, xywh
        )
        
        # Update metadata
        self.score = confidence
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        self.hit_streak += 1
        
        # Clear occlusion
        self.is_occluded = False
        self.occlusion_count = 0
        
        # Update appearance
        if appearance_feature is not None:
            self._update_appearance(appearance_feature)
        
        # Update mask
        if mask is not None:
            self.mask_history.append(mask.copy())
            self.current_mask = mask
        
        # State transition
        if self.state == TrackState.NEW:
            if self.tracklet_len >= 3:  # Confirm after 3 frames
                self.state = TrackState.TRACKED
                self.is_activated = True
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED
        
        self._update_position_history()
    
    def _update_appearance(self, feature: AppearanceFeature):
        """Update appearance feature gallery."""
        # Only add high-quality features
        if feature.quality > 0.3:
            self.appearance_history.append(feature)
        
        # Update smooth appearance (EMA)
        if self.smooth_appearance is None:
            self.smooth_appearance = feature
        else:
            alpha = 0.9  # High weight to new feature
            
            # Smooth histogram
            new_hist = alpha * feature.histogram + (1 - alpha) * self.smooth_appearance.histogram
            new_hist /= new_hist.sum() + 1e-6
            
            # Smooth deep feature
            new_deep = None
            if feature.deep_feature is not None:
                if self.smooth_appearance.deep_feature is not None:
                    new_deep = alpha * feature.deep_feature + (1 - alpha) * self.smooth_appearance.deep_feature
                    new_deep /= np.linalg.norm(new_deep) + 1e-6
                else:
                    new_deep = feature.deep_feature
            
            self.smooth_appearance = AppearanceFeature(
                histogram=new_hist,
                deep_feature=new_deep,
                quality=max(feature.quality, self.smooth_appearance.quality),
                frame_id=feature.frame_id
            )
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST
        self.hit_streak = 0
    
    def mark_removed(self):
        """Mark track for removal."""
        self.state = TrackState.REMOVED
    
    def activate(self, frame_id: int):
        """Activate track."""
        self.is_activated = True
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(
        self,
        detection: Tuple[np.ndarray, float, int],
        frame_id: int,
        new_id: bool = False
    ):
        """
        Re-activate a lost track.
        
        Args:
            detection: New detection
            frame_id: Current frame
            new_id: Whether to assign new ID
        """
        if new_id:
            EnhancedTrack._count += 1
            self.track_id = EnhancedTrack._count
        
        tlwh, confidence, class_id = detection
        
        # Re-initialize UKF with new detection
        xywh = self._tlwh_to_xywh(tlwh)
        ukf = self.get_ukf()
        self.mean, self.covariance = ukf.initiate(xywh)
        
        # Preserve velocity from last state if available
        if len(self.position_history) >= 2:
            recent_positions = list(self.position_history)[-5:]
            if len(recent_positions) >= 2:
                velocity = (recent_positions[-1] - recent_positions[0]) / len(recent_positions)
                self.mean[4:6] = velocity  # Set vx, vy
        
        self.score = confidence
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.time_since_update = 0
        self.hit_streak = 1
        self.is_occluded = False
        self.occlusion_count = 0
    
    def get_memory(self) -> TrackMemory:
        """Get track memory for re-identification."""
        bbox = self.tlwh
        position = bbox[:2] + bbox[2:] / 2  # Center
        velocity = self.mean[4:6] if len(self.mean) > 5 else np.zeros(2)
        
        return TrackMemory(
            track_id=self.track_id,
            class_id=self.class_id,
            last_bbox=bbox.copy(),
            last_position=position,
            last_velocity=velocity,
            appearance_features=list(self.appearance_history),
            mask_features=list(self.mask_history),
            lost_frame=self.frame_id,
            confidence=self.score
        )
    
    @property
    def tlwh(self) -> np.ndarray:
        """Get current bbox in [x1, y1, w, h] format."""
        bbox = self.get_ukf().state_to_bbox(self.mean)
        return self._xywh_to_tlwh(bbox)
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get current bbox in [x1, y1, x2, y2] format."""
        tlwh = self.tlwh
        tlbr = tlwh.copy()
        tlbr[2:] += tlwh[:2]
        return tlbr
    
    @property
    def xywh(self) -> np.ndarray:
        """Get current bbox in [cx, cy, w, h] format."""
        return self.get_ukf().state_to_bbox(self.mean)
    
    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return self.mean[4:6]
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.TRACKED
    
    @property
    def end_frame(self) -> int:
        """Get last frame where track was updated."""
        return self.frame_id
    
    def get_appearance_distance(self, feature: AppearanceFeature) -> float:
        """
        Compute appearance distance to this track.
        
        Args:
            feature: Detection appearance feature
            
        Returns:
            Distance (0 = identical, 1 = different)
        """
        if self.smooth_appearance is None:
            return 1.0
        
        import cv2
        
        # Histogram distance
        hist_dist = cv2.compareHist(
            feature.histogram.astype(np.float32),
            self.smooth_appearance.histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        # Deep feature distance
        if (feature.deep_feature is not None and 
            self.smooth_appearance.deep_feature is not None):
            cosine_sim = np.dot(
                feature.deep_feature, 
                self.smooth_appearance.deep_feature
            )
            deep_dist = 1.0 - cosine_sim
            return 0.3 * hist_dist + 0.7 * deep_dist
        
        return hist_dist


# ============================================================================
# Utility Functions for Matching
# ============================================================================

def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bboxes.
    
    Args:
        bboxes1: [N, 4] in tlbr format
        bboxes2: [M, 4] in tlbr format
        
    Returns:
        IoU matrix [N, M]
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    
    # Compute intersection
    x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0].T)
    y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1].T)
    x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2].T)
    y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3].T)
    
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    
    union_area = area1[:, np.newaxis] + area2 - inter_area
    
    return inter_area / (union_area + 1e-6)


def mask_iou_batch(masks1: List[np.ndarray], masks2: List[np.ndarray]) -> np.ndarray:
    """
    Compute IoU between two sets of masks.
    
    Args:
        masks1: List of N binary masks
        masks2: List of M binary masks
        
    Returns:
        IoU matrix [N, M]
    """
    n, m = len(masks1), len(masks2)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    iou_matrix = np.zeros((n, m))
    
    for i, m1 in enumerate(masks1):
        for j, m2 in enumerate(masks2):
            if m1 is None or m2 is None:
                continue
            if m1.shape != m2.shape:
                continue
            
            intersection = np.logical_and(m1 > 0, m2 > 0).sum()
            union = np.logical_or(m1 > 0, m2 > 0).sum()
            
            if union > 0:
                iou_matrix[i, j] = intersection / union
    
    return iou_matrix


def linear_assignment(cost_matrix: np.ndarray, threshold: float = 0.8):
    """
    Solve linear assignment problem.
    
    Args:
        cost_matrix: Cost matrix [N, M]
        threshold: Maximum cost for valid assignment
        
    Returns:
        matches: List of (track_idx, det_idx) pairs
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        x = list(x)
        y = list(y)
    
    matches = []
    unmatched_tracks = []
    unmatched_dets = []
    
    for i in range(cost_matrix.shape[0]):
        if isinstance(x, list):
            if i < len(x) and cost_matrix[i, x[i]] <= threshold:
                matches.append((i, x[i]))
            else:
                unmatched_tracks.append(i)
        else:
            if x[i] >= 0 and cost_matrix[i, x[i]] <= threshold:
                matches.append((i, x[i]))
            else:
                unmatched_tracks.append(i)
    
    matched_det_indices = {m[1] for m in matches}
    for j in range(cost_matrix.shape[1]):
        if j not in matched_det_indices:
            unmatched_dets.append(j)
    
    return matches, unmatched_tracks, unmatched_dets


def fuse_motion_appearance(
    motion_cost: np.ndarray,
    appearance_cost: np.ndarray,
    motion_weight: float = 0.5
) -> np.ndarray:
    """
    Fuse motion and appearance costs.
    
    Args:
        motion_cost: Motion-based cost matrix (1 - IoU)
        appearance_cost: Appearance-based cost matrix
        motion_weight: Weight for motion cost
        
    Returns:
        Fused cost matrix
    """
    appearance_weight = 1.0 - motion_weight
    return motion_weight * motion_cost + appearance_weight * appearance_cost
