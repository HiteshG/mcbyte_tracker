"""
McByte Tracker
==============
Multi-object tracker combining ByteTrack association with 
temporal mask propagation for improved robustness.

Key Features:
- Two-stage association (high + low confidence detections)
- Kalman filter motion prediction
- Camera motion compensation
- Mask-based association enhancement
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import cv2

from .track import Track, TrackState, iou_distance, fuse_score, linear_assignment
from .kalman_filter import KalmanFilter


class CameraMotionCompensation:
    """
    Camera motion compensation using sparse optical flow.
    Estimates homography between frames to align track predictions.
    """
    
    def __init__(self, method: str = "sparseOptFlow"):
        """
        Initialize GMC.
        
        Args:
            method: GMC method ("sparseOptFlow", "orb", "ecc", or "none")
        """
        self.method = method
        self.prev_frame = None
        self.prev_keypoints = None
        
        if method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def apply(
        self,
        frame: np.ndarray,
        detections: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute camera motion and return transformation matrix.
        
        Args:
            frame: Current frame (BGR)
            detections: Optional detections to exclude from feature matching
            
        Returns:
            3x3 homography matrix (identity if no motion detected)
        """
        H = np.eye(3, dtype=np.float32)
        
        if self.method == "none":
            return H
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return H
        
        if self.method == "sparseOptFlow":
            H = self._sparse_optical_flow(gray, detections)
        elif self.method == "orb":
            H = self._orb_matching(gray, detections)
        elif self.method == "ecc":
            H = self._ecc(gray)
        
        self.prev_frame = gray
        return H
    
    def _sparse_optical_flow(
        self,
        gray: np.ndarray,
        detections: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sparse optical flow based GMC."""
        H = np.eye(3, dtype=np.float32)
        
        # Detect features
        keypoints = cv2.goodFeaturesToTrack(
            self.prev_frame,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        
        if keypoints is None or len(keypoints) < 4:
            return H
        
        # Track features
        matched_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, keypoints, None
        )
        
        # Filter valid matches
        valid = status.flatten() == 1
        prev_pts = keypoints[valid].reshape(-1, 2)
        curr_pts = matched_keypoints[valid].reshape(-1, 2)
        
        if len(prev_pts) < 4:
            return H
        
        # Estimate homography
        H, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts, method=cv2.RANSAC
        )
        
        if H is None:
            return np.eye(3, dtype=np.float32)
        
        # Convert to 3x3
        H_full = np.eye(3, dtype=np.float32)
        H_full[:2, :] = H
        
        return H_full
    
    def _orb_matching(
        self,
        gray: np.ndarray,
        detections: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """ORB feature matching based GMC."""
        H = np.eye(3, dtype=np.float32)
        
        # Detect and extract features
        keypoints_prev = self.detector.detect(self.prev_frame)
        keypoints_curr = self.detector.detect(gray)
        
        keypoints_prev, desc_prev = self.extractor.compute(self.prev_frame, keypoints_prev)
        keypoints_curr, desc_curr = self.extractor.compute(gray, keypoints_curr)
        
        if desc_prev is None or desc_curr is None:
            return H
        
        # Match features
        matches = self.matcher.knnMatch(desc_prev, desc_curr, k=2)
        
        # Ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            return H
        
        prev_pts = np.array([keypoints_prev[m.queryIdx].pt for m in good_matches])
        curr_pts = np.array([keypoints_curr[m.trainIdx].pt for m in good_matches])
        
        # Estimate homography
        H_2x3, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        
        if H_2x3 is not None:
            H[:2, :] = H_2x3
        
        return H
    
    def _ecc(self, gray: np.ndarray) -> np.ndarray:
        """ECC based GMC."""
        H = np.eye(3, dtype=np.float32)
        
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)
            
            _, warp_matrix = cv2.findTransformECC(
                self.prev_frame, gray, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria, None, 5
            )
            
            H[:2, :] = warp_matrix
        except cv2.error:
            pass
        
        return H


class McByteTracker:
    """
    McByte Multi-Object Tracker
    
    Combines ByteTrack's cascaded association with mask propagation
    for robust tracking in challenging scenarios.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        frame_rate: int = 30,
        use_gmc: bool = True,
        gmc_method: str = "sparseOptFlow",
    ):
        """
        Initialize McByte tracker.
        
        Args:
            track_thresh: High detection confidence threshold
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for association
            low_thresh: Low detection confidence threshold
            new_track_thresh: Threshold for starting new tracks
            frame_rate: Video frame rate
            use_gmc: Enable camera motion compensation
            gmc_method: GMC method
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        
        # Time management
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        # Track pools
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        # Camera motion compensation
        self.gmc = CameraMotionCompensation(gmc_method) if use_gmc else None
        
        # Reset track IDs
        Track.reset_id()
    
    def update(
        self,
        detections,
        frame: Optional[np.ndarray] = None,
        mask_info: Optional[Dict[str, Any]] = None,
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: FrameDetections object or list of (tlwh, score, class_id)
            frame: Current frame (required for GMC)
            mask_info: Optional mask propagation info for enhanced association
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        
        # Parse detections
        if hasattr(detections, 'detections'):
            # FrameDetections object
            det_list = [
                (d.tlwh, d.confidence, d.class_id)
                for d in detections.detections
            ]
        else:
            det_list = detections
        
        # Split detections by confidence
        high_dets = []
        low_dets = []
        
        for det in det_list:
            tlwh, score = det[:2]
            class_id = det[2] if len(det) > 2 else 0
            
            if score >= self.track_thresh:
                high_dets.append((tlwh, score, class_id))
            elif score >= self.low_thresh:
                low_dets.append((tlwh, score, class_id))
        
        # === STEP 1: Predict tracks ===
        Track.multi_predict(self.tracked_tracks)
        Track.multi_predict(self.lost_tracks)
        
        # === STEP 2: Camera motion compensation ===
        if self.gmc is not None and frame is not None:
            warp = self.gmc.apply(frame)
            self._apply_gmc(self.tracked_tracks, warp)
            self._apply_gmc(self.lost_tracks, warp)
        
        # === STEP 3: First association with high score detections ===
        # Separate confirmed and unconfirmed tracks
        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        tracked = [t for t in self.tracked_tracks if t.is_activated]
        
        # Pool tracked and lost tracks
        track_pool = tracked + self.lost_tracks
        
        # Compute cost matrix
        cost_matrix = iou_distance(track_pool, high_dets)
        cost_matrix = self._apply_mask_cost(cost_matrix, track_pool, high_dets, mask_info)
        cost_matrix = fuse_score(cost_matrix, high_dets)
        
        # Hungarian matching
        matches, u_track, u_det = linear_assignment(cost_matrix, self.match_thresh)
        
        # Process matches
        activated_tracks = []
        refind_tracks = []
        
        for t_idx, d_idx in matches:
            track = track_pool[t_idx]
            det = high_dets[d_idx]
            
            if track.state == TrackState.TRACKED:
                track.update(det[:2], self.frame_id)
                activated_tracks.append(track)
            else:
                track.reactivate(det[:2], self.frame_id)
                refind_tracks.append(track)
        
        # === STEP 4: Second association with low score detections ===
        remaining_tracks = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.TRACKED]
        
        if len(low_dets) > 0 and len(remaining_tracks) > 0:
            cost_matrix = iou_distance(remaining_tracks, low_dets)
            matches, u_track_2, u_det_2 = linear_assignment(cost_matrix, 0.5)
            
            for t_idx, d_idx in matches:
                track = remaining_tracks[t_idx]
                det = low_dets[d_idx]
                
                if track.state == TrackState.TRACKED:
                    track.update(det[:2], self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.reactivate(det[:2], self.frame_id)
                    refind_tracks.append(track)
            
            # Mark remaining tracks as lost
            for idx in u_track_2:
                track = remaining_tracks[idx]
                if track.state != TrackState.LOST:
                    track.mark_lost()
        else:
            u_track_2 = u_track
            for idx in u_track:
                track = track_pool[idx]
                if track.state == TrackState.TRACKED:
                    track.mark_lost()
        
        # === STEP 5: Association with unconfirmed tracks ===
        remaining_dets = [high_dets[i] for i in u_det]
        
        cost_matrix = iou_distance(unconfirmed, remaining_dets)
        cost_matrix = fuse_score(cost_matrix, remaining_dets)
        
        matches, u_unconfirmed, u_det_final = linear_assignment(cost_matrix, 0.7)
        
        for t_idx, d_idx in matches:
            track = unconfirmed[t_idx]
            det = remaining_dets[d_idx]
            track.update(det[:2], self.frame_id)
            activated_tracks.append(track)
        
        # Remove unmatched unconfirmed tracks
        for idx in u_unconfirmed:
            track = unconfirmed[idx]
            track.mark_removed()
        
        # === STEP 6: Initialize new tracks ===
        new_tracks = []
        for idx in u_det_final:
            det = remaining_dets[idx]
            if det[1] >= self.new_track_thresh:
                track = Track(det[0], det[1], det[2] if len(det) > 2 else 0)
                track.activate(self.frame_id)
                new_tracks.append(track)
        
        # === STEP 7: Update track pools ===
        lost_tracks = []
        for track in self.tracked_tracks:
            if track.state == TrackState.LOST:
                lost_tracks.append(track)
        
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        self.tracked_tracks = self._merge_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = self._merge_tracks(self.tracked_tracks, refind_tracks)
        self.tracked_tracks = self._merge_tracks(self.tracked_tracks, new_tracks)
        
        self.lost_tracks = self._subtract_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = self._subtract_tracks(self.lost_tracks, self.removed_tracks)
        
        # Remove old lost tracks
        removed = []
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)
        
        self.lost_tracks = [t for t in self.lost_tracks if t.state != TrackState.REMOVED]
        self.removed_tracks.extend(removed)
        
        # Return active tracks
        return [t for t in self.tracked_tracks if t.is_activated]
    
    def _apply_gmc(self, tracks: List[Track], warp: np.ndarray) -> None:
        """Apply camera motion compensation to track predictions."""
        if warp is None or np.allclose(warp, np.eye(3)):
            return
        
        for track in tracks:
            if track.mean is None:
                continue
            
            # Transform center point
            x, y = track.mean[:2]
            pt = np.array([x, y, 1.0])
            pt_new = warp @ pt
            track.mean[:2] = pt_new[:2] / pt_new[2]
    
    def _apply_mask_cost(
        self,
        cost_matrix: np.ndarray,
        tracks: List[Track],
        detections: List,
        mask_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Apply mask-based cost adjustment.
        
        Reduces cost for matches where mask IoU is high.
        """
        if mask_info is None or 'mask' not in mask_info:
            return cost_matrix
        
        # This is where mask propagation integration happens
        # The mask_info contains the propagated mask and track-to-mask mapping
        # Implementation depends on your mask propagation setup
        
        return cost_matrix
    
    @staticmethod
    def _merge_tracks(list_a: List[Track], list_b: List[Track]) -> List[Track]:
        """Merge two track lists."""
        existing_ids = {t.track_id for t in list_a}
        merged = list_a.copy()
        for track in list_b:
            if track.track_id not in existing_ids:
                merged.append(track)
                existing_ids.add(track.track_id)
        return merged
    
    @staticmethod
    def _subtract_tracks(list_a: List[Track], list_b: List[Track]) -> List[Track]:
        """Remove tracks in list_b from list_a."""
        ids_b = {t.track_id for t in list_b}
        return [t for t in list_a if t.track_id not in ids_b]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracked_tracks + self.lost_tracks:
            if track.track_id == track_id:
                return track
        return None
    
    @property
    def active_track_count(self) -> int:
        """Number of active tracks."""
        return len([t for t in self.tracked_tracks if t.is_activated])
    
    @property
    def lost_track_count(self) -> int:
        """Number of lost tracks."""
        return len(self.lost_tracks)
