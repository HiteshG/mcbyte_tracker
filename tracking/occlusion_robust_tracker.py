"""
Occlusion-Robust Hockey Tracker
================================
Production-ready tracker with full identity preservation through occlusions.

Integrates:
- Unscented Kalman Filter for non-linear state estimation
- SAM for high-quality mask generation
- CUTIE for temporal mask propagation
- Appearance features for re-identification
- Multi-cue association (motion + appearance + mask)
- Track memory for lost track recovery

This is the main tracker class to use for hockey tracking.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from .enhanced_track import (
    EnhancedTrack, TrackState, TrackMemory,
    iou_batch, linear_assignment, fuse_motion_appearance
)
from .appearance import AppearanceExtractor, AppearanceFeature
from ..mask_propagation.integrated_masks import (
    MaskPropagationSystem, MaskIdentityVerifier, PropagationResult
)


@dataclass
class TrackerConfig:
    """Configuration for OcclusionRobustTracker."""
    # Detection thresholds
    track_high_thresh: float = 0.5      # High confidence detection threshold
    track_low_thresh: float = 0.1       # Low confidence detection threshold
    new_track_thresh: float = 0.6       # Threshold for creating new tracks
    
    # Matching thresholds
    match_thresh: float = 0.8           # IoU matching threshold
    appearance_thresh: float = 0.4      # Appearance matching threshold
    mask_thresh: float = 0.3            # Mask IoU threshold
    
    # Track lifecycle
    track_buffer: int = 60              # Frames to keep lost tracks (2 sec at 30fps)
    confirm_frames: int = 3             # Frames to confirm new track
    
    # Feature weights
    motion_weight: float = 0.4          # Weight for motion cost
    appearance_weight: float = 0.35     # Weight for appearance cost
    mask_weight: float = 0.25           # Weight for mask cost
    
    # Camera motion compensation
    use_gmc: bool = True
    gmc_method: str = "sparse"
    
    # Components
    use_appearance: bool = True
    use_masks: bool = True
    use_reid: bool = True               # Re-identification for lost tracks
    
    # Device
    device: str = "cuda"
    
    # SAM/CUTIE paths (optional, will use defaults if available)
    sam_checkpoint: Optional[str] = None
    cutie_checkpoint: Optional[str] = None


@dataclass
class FrameResult:
    """Result for a single frame."""
    tracks: List[EnhancedTrack]
    masks: Dict[int, np.ndarray]        # track_id -> mask
    combined_mask: Optional[np.ndarray]  # All masks with IDs
    timing: Dict[str, float]            # Component timings


class OcclusionRobustTracker:
    """
    Main tracker class for hockey with full occlusion handling.
    
    Usage:
        tracker = OcclusionRobustTracker(TrackerConfig())
        
        for frame in video:
            detections = detector.detect(frame)
            result = tracker.update(detections, frame)
            
            for track in result.tracks:
                print(f"Track {track.track_id}: {track.tlbr}")
    """
    
    def __init__(self, config: TrackerConfig = None):
        """
        Initialize tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        
        # Track storage
        self.tracked_tracks: List[EnhancedTrack] = []
        self.lost_tracks: List[EnhancedTrack] = []
        self.removed_tracks: List[EnhancedTrack] = []
        
        # Track memory for re-identification
        self.track_memory: Dict[int, TrackMemory] = {}
        self.memory_max_age = self.config.track_buffer
        
        # Frame counter
        self.frame_id = 0
        
        # Appearance extractor
        self.appearance_extractor = None
        if self.config.use_appearance:
            self.appearance_extractor = AppearanceExtractor(
                use_deep_features=self.config.use_reid,
                device=self.config.device
            )
        
        # Mask propagation system
        self.mask_system = None
        self.mask_verifier = None
        if self.config.use_masks:
            self.mask_system = MaskPropagationSystem(
                sam_checkpoint=self.config.sam_checkpoint,
                cutie_checkpoint=self.config.cutie_checkpoint,
                device=self.config.device
            )
            self.mask_verifier = MaskIdentityVerifier(self.mask_system)
        
        # Camera motion compensation
        self.gmc = None
        if self.config.use_gmc:
            self.gmc = CameraMotionCompensator(method=self.config.gmc_method)
        
        # Reset track ID counter
        EnhancedTrack.reset_id_counter()
    
    def update(
        self,
        detections: List[Tuple[np.ndarray, float, int]],
        frame: np.ndarray
    ) -> FrameResult:
        """
        Process new frame and update tracks.
        
        Args:
            detections: List of (tlwh, confidence, class_id)
            frame: Current frame (BGR)
            
        Returns:
            FrameResult with tracks and masks
        """
        import time
        timing = {}
        
        self.frame_id += 1
        
        # ============================
        # 1. CAMERA MOTION COMPENSATION
        # ============================
        t0 = time.time()
        if self.gmc is not None:
            warp_matrix = self.gmc.compute(frame)
            if warp_matrix is not None:
                self._apply_gmc(warp_matrix)
        timing['gmc'] = time.time() - t0
        
        # ============================
        # 2. PREDICT TRACK STATES
        # ============================
        t0 = time.time()
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        timing['predict'] = time.time() - t0
        
        # ============================
        # 3. MASK PROPAGATION
        # ============================
        t0 = time.time()
        mask_result = PropagationResult(masks={}, confidences={})
        if self.mask_system is not None:
            active_ids = {t.track_id for t in self.tracked_tracks}
            mask_result = self.mask_system.propagate(frame, active_ids)
        timing['masks'] = time.time() - t0
        
        # ============================
        # 4. EXTRACT DETECTION FEATURES
        # ============================
        t0 = time.time()
        det_features = []
        det_masks = []
        
        for det in detections:
            tlwh, conf, cls_id = det
            bbox = np.array([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]])
            
            # Extract appearance feature
            feat = None
            if self.appearance_extractor is not None:
                feat = self.appearance_extractor.extract(
                    frame, bbox, frame_id=self.frame_id
                )
            det_features.append(feat)
            
            # Generate mask for detection (if needed for matching)
            det_mask = None
            if self.mask_system is not None and self.config.use_masks:
                # Only generate masks for high-confidence detections
                if conf >= self.config.track_high_thresh:
                    det_mask = self.mask_system._regenerate_mask(frame, -1, bbox)
            det_masks.append(det_mask)
        
        timing['features'] = time.time() - t0
        
        # ============================
        # 5. SPLIT DETECTIONS BY CONFIDENCE
        # ============================
        high_dets = []
        high_feats = []
        high_masks = []
        low_dets = []
        low_feats = []
        
        for i, (det, feat, mask) in enumerate(zip(detections, det_features, det_masks)):
            tlwh, conf, cls_id = det
            if conf >= self.config.track_high_thresh:
                high_dets.append(det)
                high_feats.append(feat)
                high_masks.append(mask)
            elif conf >= self.config.track_low_thresh:
                low_dets.append(det)
                low_feats.append(feat)
        
        # ============================
        # 6. CASCADED ASSOCIATION
        # ============================
        t0 = time.time()
        
        # Separate tracks
        confirmed_tracks = [t for t in self.tracked_tracks if t.is_confirmed]
        unconfirmed_tracks = [t for t in self.tracked_tracks if not t.is_confirmed]
        
        # STAGE 1: Match confirmed tracks with high-confidence detections
        matches1, u_tracks1, u_dets1 = self._match_tracks_detections(
            confirmed_tracks, high_dets, high_feats, high_masks,
            thresh=self.config.match_thresh
        )
        
        # STAGE 2: Match remaining confirmed + lost tracks with remaining high detections
        remaining_tracks = [confirmed_tracks[i] for i in u_tracks1] + self.lost_tracks
        remaining_dets = [high_dets[i] for i in u_dets1]
        remaining_feats = [high_feats[i] for i in u_dets1]
        remaining_masks = [high_masks[i] for i in u_dets1]
        
        matches2, u_tracks2, u_dets2 = self._match_tracks_detections(
            remaining_tracks, remaining_dets, remaining_feats, remaining_masks,
            thresh=self.config.match_thresh + 0.1  # Slightly higher threshold for recovery
        )
        
        # STAGE 3: Match unconfirmed tracks with remaining detections
        remaining_dets3 = [remaining_dets[i] for i in u_dets2]
        remaining_feats3 = [remaining_feats[i] for i in u_dets2]
        remaining_masks3 = [remaining_masks[i] for i in u_dets2]
        
        matches3, u_tracks3, u_dets3 = self._match_tracks_detections(
            unconfirmed_tracks, remaining_dets3, remaining_feats3, remaining_masks3,
            thresh=0.7, use_appearance=False  # Simpler matching for unconfirmed
        )
        
        # STAGE 4: Match remaining tracks with low-confidence detections
        still_unmatched = []
        for i in u_tracks2:
            if i < len([t for t in remaining_tracks if t in confirmed_tracks]):
                still_unmatched.append(remaining_tracks[i])
        
        matches4 = []
        if len(still_unmatched) > 0 and len(low_dets) > 0:
            matches4, _, _ = self._match_tracks_detections(
                still_unmatched, low_dets, low_feats, [],
                thresh=0.5, use_appearance=False, use_masks=False
            )
        
        timing['association'] = time.time() - t0
        
        # ============================
        # 7. UPDATE TRACKS
        # ============================
        t0 = time.time()
        
        activated_tracks = []
        refound_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Process Stage 1 matches
        for t_idx, d_idx in matches1:
            track = confirmed_tracks[t_idx]
            det = high_dets[d_idx]
            feat = high_feats[d_idx]
            mask = high_masks[d_idx]
            
            track.update(det, self.frame_id, feat, mask)
            
            if feat is not None and self.appearance_extractor is not None:
                self.appearance_extractor.update_gallery(
                    track.track_id, feat, self.frame_id
                )
            
            if mask is not None and self.mask_system is not None:
                self.mask_system.update_mask(frame, track.track_id, track.tlbr)
            
            activated_tracks.append(track)
        
        # Process Stage 2 matches (re-found tracks)
        for t_idx, d_idx in matches2:
            track = remaining_tracks[t_idx]
            det = remaining_dets[d_idx]
            feat = remaining_feats[d_idx]
            mask = remaining_masks[d_idx]
            
            if track.state == TrackState.LOST:
                track.re_activate(det, self.frame_id, new_id=False)
                refound_tracks.append(track)
            else:
                track.update(det, self.frame_id, feat, mask)
                activated_tracks.append(track)
            
            if feat is not None and self.appearance_extractor is not None:
                self.appearance_extractor.update_gallery(
                    track.track_id, feat, self.frame_id
                )
        
        # Process Stage 3 matches (unconfirmed)
        for t_idx, d_idx in matches3:
            track = unconfirmed_tracks[t_idx]
            det = remaining_dets3[d_idx]
            feat = remaining_feats3[d_idx] if d_idx < len(remaining_feats3) else None
            
            track.update(det, self.frame_id, feat)
            activated_tracks.append(track)
        
        # Process Stage 4 matches (low confidence)
        for t_idx, d_idx in matches4:
            track = still_unmatched[t_idx]
            det = low_dets[d_idx]
            track.update(det, self.frame_id)
            activated_tracks.append(track)
        
        # Handle unmatched tracks
        for i in u_tracks1:
            if i not in [m[0] for m in matches1]:
                track = confirmed_tracks[i]
                if track not in [remaining_tracks[j] for j, _ in matches2]:
                    if track.time_since_update <= self.config.track_buffer:
                        track.mark_lost()
                        lost_tracks.append(track)
                        self._store_track_memory(track)
                    else:
                        track.mark_removed()
                        removed_tracks.append(track)
        
        # Handle unmatched unconfirmed tracks
        for i in u_tracks3:
            track = unconfirmed_tracks[i]
            track.mark_removed()
            removed_tracks.append(track)
        
        # Handle unmatched lost tracks
        for track in self.lost_tracks:
            if track not in [remaining_tracks[j] for j, _ in matches2]:
                if track.time_since_update > self.config.track_buffer:
                    track.mark_removed()
                    removed_tracks.append(track)
                else:
                    lost_tracks.append(track)
        
        # Create new tracks from unmatched high-confidence detections
        final_unmatched_dets = []
        # Collect all truly unmatched detection indices
        matched_det_indices = set()
        for _, d_idx in matches1:
            matched_det_indices.add(d_idx)
        for _, d_idx in matches2:
            matched_det_indices.add(u_dets1[d_idx])
        for _, d_idx in matches3:
            matched_det_indices.add(u_dets1[u_dets2[d_idx]])
        
        for i, det in enumerate(high_dets):
            if i not in matched_det_indices:
                tlwh, conf, cls_id = det
                if conf >= self.config.new_track_thresh:
                    feat = high_feats[i]
                    track = EnhancedTrack(det, self.frame_id, feat)
                    
                    # Initialize mask for new track
                    if self.mask_system is not None:
                        self.mask_system.initialize_track_mask(
                            frame, track.track_id, track.tlbr, cls_id
                        )
                    
                    activated_tracks.append(track)
        
        timing['update'] = time.time() - t0
        
        # ============================
        # 8. FINALIZE
        # ============================
        
        # Update track lists
        self.tracked_tracks = [t for t in activated_tracks + refound_tracks 
                              if t.state == TrackState.TRACKED or t.state == TrackState.NEW]
        self.lost_tracks = [t for t in lost_tracks if t.state == TrackState.LOST]
        self.removed_tracks.extend(removed_tracks)
        
        # Cleanup
        if self.appearance_extractor is not None:
            active_ids = {t.track_id for t in self.tracked_tracks + self.lost_tracks}
            self.appearance_extractor.cleanup(active_ids)
        
        # Clean old track memory
        self._cleanup_memory()
        
        # Get output tracks (only confirmed)
        output_tracks = [t for t in self.tracked_tracks if t.is_confirmed]
        
        # Get masks for output tracks
        output_masks = {}
        for track in output_tracks:
            if self.mask_system is not None:
                mask = self.mask_system.get_mask(track.track_id)
                if mask is not None:
                    output_masks[track.track_id] = mask
        
        return FrameResult(
            tracks=output_tracks,
            masks=output_masks,
            combined_mask=mask_result.combined_mask,
            timing=timing
        )
    
    def _match_tracks_detections(
        self,
        tracks: List[EnhancedTrack],
        detections: List[Tuple[np.ndarray, float, int]],
        features: List[Optional[AppearanceFeature]],
        masks: List[Optional[np.ndarray]],
        thresh: float = 0.8,
        use_appearance: bool = True,
        use_masks: bool = True
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks to detections using multiple cues.
        
        Returns:
            matches, unmatched_tracks, unmatched_detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Get track and detection bboxes
        track_bboxes = np.array([t.tlbr for t in tracks])
        det_bboxes = np.array([
            [d[0][0], d[0][1], d[0][0]+d[0][2], d[0][1]+d[0][3]]
            for d in detections
        ])
        
        # 1. Motion cost (1 - IoU)
        iou_matrix = iou_batch(track_bboxes, det_bboxes)
        motion_cost = 1.0 - iou_matrix
        
        # 2. Appearance cost
        appearance_cost = np.ones_like(motion_cost)
        if use_appearance and self.config.use_appearance and self.appearance_extractor is not None:
            for i, track in enumerate(tracks):
                for j, feat in enumerate(features):
                    if feat is not None:
                        appearance_cost[i, j] = track.get_appearance_distance(feat)
        
        # 3. Mask cost
        mask_cost = np.ones_like(motion_cost)
        if use_masks and self.config.use_masks and self.mask_system is not None:
            track_ids = [t.track_id for t in tracks]
            mask_cost = self.mask_system.compute_mask_cost_matrix(track_ids, masks)
        
        # Fuse costs
        cost_matrix = (
            self.config.motion_weight * motion_cost +
            self.config.appearance_weight * appearance_cost +
            self.config.mask_weight * mask_cost
        )
        
        # Apply gating
        cost_matrix[motion_cost > 0.8] = 1.0  # Gate by IoU
        
        # Solve assignment
        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            cost_matrix, threshold=thresh
        )
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _store_track_memory(self, track: EnhancedTrack):
        """Store track memory for re-identification."""
        memory = track.get_memory()
        self.track_memory[track.track_id] = memory
        
        # Store in mask verifier
        if self.mask_verifier is not None:
            self.mask_verifier.store_lost_track(
                track.track_id,
                track.current_mask,
                memory.last_position,
                memory.last_velocity,
                self.frame_id
            )
    
    def _cleanup_memory(self):
        """Remove old track memories."""
        to_remove = []
        for tid, memory in self.track_memory.items():
            age = self.frame_id - memory.lost_frame
            if age > self.memory_max_age:
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.track_memory[tid]
            if self.mask_verifier is not None:
                self.mask_verifier.lost_track_masks.pop(tid, None)
                self.mask_verifier.lost_track_positions.pop(tid, None)
                self.mask_verifier.lost_track_velocities.pop(tid, None)
                self.mask_verifier.lost_frame.pop(tid, None)
    
    def _apply_gmc(self, warp_matrix: np.ndarray):
        """Apply camera motion compensation to tracks."""
        for track in self.tracked_tracks + self.lost_tracks:
            # Get current position
            x, y = track.mean[:2]
            
            # Apply warp
            point = np.array([[x, y]], dtype=np.float32)
            warped = cv2.transform(point.reshape(-1, 1, 2), warp_matrix[:2, :])
            
            # Update position
            track.mean[0] = warped[0, 0, 0]
            track.mean[1] = warped[0, 0, 1]
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.track_memory = {}
        self.frame_id = 0
        
        EnhancedTrack.reset_id_counter()
        
        if self.mask_system is not None:
            self.mask_system.reset()
        
        if self.gmc is not None:
            self.gmc.reset()


class CameraMotionCompensator:
    """
    Camera Motion Compensation using various methods.
    Essential for hockey with moving cameras.
    """
    
    def __init__(self, method: str = "sparse"):
        """
        Args:
            method: "sparse" (optical flow), "orb" (feature matching), "ecc"
        """
        self.method = method
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # ORB detector for feature-based
        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Optical flow params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def compute(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute camera motion warp matrix.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            3x3 warp matrix or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            if self.method == "sparse":
                self.prev_keypoints = cv2.goodFeaturesToTrack(
                    gray, maxCorners=200, qualityLevel=0.01, minDistance=30
                )
            return None
        
        warp_matrix = None
        
        if self.method == "sparse":
            warp_matrix = self._compute_sparse(gray)
        elif self.method == "orb":
            warp_matrix = self._compute_orb(gray)
        elif self.method == "ecc":
            warp_matrix = self._compute_ecc(gray)
        
        self.prev_frame = gray
        
        return warp_matrix
    
    def _compute_sparse(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Sparse optical flow based GMC."""
        if self.prev_keypoints is None or len(self.prev_keypoints) < 4:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            return None
        
        # Track keypoints
        curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_keypoints, None, **self.lk_params
        )
        
        if curr_keypoints is None:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            return None
        
        # Filter good points
        status = status.flatten()
        good_old = self.prev_keypoints[status == 1]
        good_new = curr_keypoints[status == 1]
        
        if len(good_old) < 4:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            return None
        
        # Estimate affine transform
        try:
            warp_matrix, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            
            if warp_matrix is None:
                return None
            
            # Convert to 3x3
            warp_3x3 = np.eye(3, dtype=np.float32)
            warp_3x3[:2, :] = warp_matrix
            
        except cv2.error:
            return None
        
        # Update keypoints
        self.prev_keypoints = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=30
        )
        
        return warp_3x3
    
    def _compute_orb(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """ORB feature based GMC."""
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if (self.prev_descriptors is None or 
            descriptors is None or 
            len(keypoints) < 4):
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        
        if len(matches) < 4:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        # Get matched points
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate transform
        try:
            warp_matrix, _ = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts, method=cv2.RANSAC
            )
            
            if warp_matrix is None:
                return None
            
            warp_3x3 = np.eye(3, dtype=np.float32)
            warp_3x3[:2, :] = warp_matrix
            
        except cv2.error:
            return None
        
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return warp_3x3
    
    def _compute_ecc(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """ECC-based GMC (slower but more accurate)."""
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50, 0.001
        )
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                self.prev_frame, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            
            warp_3x3 = np.eye(3, dtype=np.float32)
            warp_3x3[:2, :] = warp_matrix
            
            return warp_3x3
            
        except cv2.error:
            return None
    
    def reset(self):
        """Reset GMC state."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
