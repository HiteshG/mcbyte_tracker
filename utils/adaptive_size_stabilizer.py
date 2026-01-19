"""
Adaptive Size Stabilizer
========================
Smooths bounding box sizes for better visualization quality.
Prevents jitter in displayed boxes without affecting tracking accuracy.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class AdaptiveSizeStabilizer:
    """
    Stabilizes bounding box sizes using motion-aware smoothing.
    
    Features:
    - Position smoothing that adapts to motion
    - Size smoothing that preserves aspect ratio
    - Velocity estimation for predictive smoothing
    - Handles rapid size changes gracefully
    """
    
    def __init__(
        self,
        history_window: int = 15,
        position_smoothing: float = 0.3,
        size_smoothing_base: float = 0.1,
        motion_threshold: float = 10.0,
        aspect_ratio_tolerance: float = 0.2
    ):
        """
        Initialize stabilizer.
        
        Args:
            history_window: Number of frames to keep in history
            position_smoothing: Smoothing factor for position (higher = more responsive)
            size_smoothing_base: Base smoothing for size (lower = more stable)
            motion_threshold: Pixel threshold for "fast motion" detection
            aspect_ratio_tolerance: Maximum allowed aspect ratio deviation
        """
        self.history_window = history_window
        self.position_smoothing = position_smoothing
        self.size_smoothing_base = size_smoothing_base
        self.motion_threshold = motion_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        
        # Per-tracker state
        self.position_history: Dict[int, deque] = {}
        self.size_history: Dict[int, deque] = {}
        self.aspect_ratio_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}
        
        # Smoothed values
        self.smooth_positions: Dict[int, np.ndarray] = {}
        self.smooth_sizes: Dict[int, np.ndarray] = {}
    
    def update(
        self,
        tracker_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Update and return smoothed bounding box.
        
        Args:
            tracker_id: Unique tracker ID
            bbox: Current bbox (x1, y1, x2, y2)
            confidence: Detection confidence (affects smoothing)
            
        Returns:
            Smoothed bbox (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        
        # Convert to center + size representation
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        position = np.array([cx, cy])
        size = np.array([w, h])
        aspect_ratio = w / max(h, 1e-6)
        
        # Initialize tracker if new
        if tracker_id not in self.position_history:
            self._init_tracker(tracker_id, position, size, aspect_ratio)
            return bbox
        
        # Compute velocity
        velocity = self._compute_velocity(tracker_id, position)
        
        # Adaptive smoothing based on motion
        pos_alpha, size_alpha = self._compute_adaptive_alphas(
            tracker_id, velocity, confidence
        )
        
        # Smooth position
        smooth_pos = self._smooth_position(tracker_id, position, pos_alpha)
        
        # Smooth size with aspect ratio preservation
        smooth_size = self._smooth_size(
            tracker_id, size, aspect_ratio, size_alpha
        )
        
        # Update histories
        self._update_histories(tracker_id, position, size, aspect_ratio, velocity)
        
        # Store smoothed values
        self.smooth_positions[tracker_id] = smooth_pos
        self.smooth_sizes[tracker_id] = smooth_size
        
        # Convert back to corner format
        sx1 = smooth_pos[0] - smooth_size[0] / 2
        sy1 = smooth_pos[1] - smooth_size[1] / 2
        sx2 = smooth_pos[0] + smooth_size[0] / 2
        sy2 = smooth_pos[1] + smooth_size[1] / 2
        
        return (
            int(round(sx1)),
            int(round(sy1)),
            int(round(sx2)),
            int(round(sy2))
        )
    
    def _init_tracker(
        self,
        tracker_id: int,
        position: np.ndarray,
        size: np.ndarray,
        aspect_ratio: float
    ):
        """Initialize state for new tracker."""
        self.position_history[tracker_id] = deque(maxlen=self.history_window)
        self.size_history[tracker_id] = deque(maxlen=self.history_window)
        self.aspect_ratio_history[tracker_id] = deque(maxlen=self.history_window)
        self.velocity_history[tracker_id] = deque(maxlen=self.history_window)
        
        self.position_history[tracker_id].append(position)
        self.size_history[tracker_id].append(size)
        self.aspect_ratio_history[tracker_id].append(aspect_ratio)
        self.velocity_history[tracker_id].append(np.zeros(2))
        
        self.smooth_positions[tracker_id] = position.copy()
        self.smooth_sizes[tracker_id] = size.copy()
    
    def _compute_velocity(
        self,
        tracker_id: int,
        position: np.ndarray
    ) -> np.ndarray:
        """Compute instantaneous velocity."""
        if len(self.position_history[tracker_id]) == 0:
            return np.zeros(2)
        
        prev_pos = self.position_history[tracker_id][-1]
        return position - prev_pos
    
    def _compute_adaptive_alphas(
        self,
        tracker_id: int,
        velocity: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Compute adaptive smoothing factors based on motion.
        
        Fast motion → higher alpha (more responsive)
        Slow motion → lower alpha (more stable)
        """
        speed = np.linalg.norm(velocity)
        
        # Motion-based adjustment
        motion_factor = min(1.0, speed / self.motion_threshold)
        
        # Confidence-based adjustment
        conf_factor = 0.5 + 0.5 * confidence
        
        # Position smoothing: more responsive during motion
        pos_alpha = self.position_smoothing + (1 - self.position_smoothing) * motion_factor
        pos_alpha *= conf_factor
        
        # Size smoothing: more stable always, slightly more responsive during motion
        size_alpha = self.size_smoothing_base + 0.1 * motion_factor
        size_alpha *= conf_factor
        
        return pos_alpha, size_alpha
    
    def _smooth_position(
        self,
        tracker_id: int,
        position: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Apply exponential smoothing to position."""
        prev_smooth = self.smooth_positions[tracker_id]
        return alpha * position + (1 - alpha) * prev_smooth
    
    def _smooth_size(
        self,
        tracker_id: int,
        size: np.ndarray,
        aspect_ratio: float,
        alpha: float
    ) -> np.ndarray:
        """
        Smooth size while preserving aspect ratio.
        
        Uses median filtering for outlier rejection + EMA for smoothing.
        """
        prev_smooth = self.smooth_sizes[tracker_id]
        
        # Check for unrealistic aspect ratio change
        if len(self.aspect_ratio_history[tracker_id]) >= 3:
            median_ar = np.median(list(self.aspect_ratio_history[tracker_id]))
            ar_deviation = abs(aspect_ratio - median_ar) / max(median_ar, 0.1)
            
            if ar_deviation > self.aspect_ratio_tolerance:
                # Aspect ratio changed too much, likely detection error
                # Use previous smooth size scaled by area change
                prev_area = prev_smooth[0] * prev_smooth[1]
                new_area = size[0] * size[1]
                area_ratio = np.sqrt(new_area / max(prev_area, 1.0))
                
                # Scale previous size by area change
                size = prev_smooth * area_ratio
        
        # Apply EMA
        smooth_size = alpha * size + (1 - alpha) * prev_smooth
        
        # Ensure minimum size
        smooth_size = np.maximum(smooth_size, [10, 10])
        
        return smooth_size
    
    def _update_histories(
        self,
        tracker_id: int,
        position: np.ndarray,
        size: np.ndarray,
        aspect_ratio: float,
        velocity: np.ndarray
    ):
        """Update all history buffers."""
        self.position_history[tracker_id].append(position.copy())
        self.size_history[tracker_id].append(size.copy())
        self.aspect_ratio_history[tracker_id].append(aspect_ratio)
        self.velocity_history[tracker_id].append(velocity.copy())
    
    def cleanup_old_trackers(self, active_tracker_ids: set):
        """Remove state for inactive trackers."""
        all_ids = set(self.position_history.keys())
        to_remove = all_ids - active_tracker_ids
        
        for tracker_id in to_remove:
            self.position_history.pop(tracker_id, None)
            self.size_history.pop(tracker_id, None)
            self.aspect_ratio_history.pop(tracker_id, None)
            self.velocity_history.pop(tracker_id, None)
            self.smooth_positions.pop(tracker_id, None)
            self.smooth_sizes.pop(tracker_id, None)
    
    def get_stats(self, tracker_id: int) -> Optional[Dict]:
        """Get statistics for a tracker."""
        if tracker_id not in self.position_history:
            return None
        
        velocities = np.array(self.velocity_history[tracker_id])
        sizes = np.array(self.size_history[tracker_id])
        
        return {
            "avg_speed": np.mean(np.linalg.norm(velocities, axis=1)),
            "max_speed": np.max(np.linalg.norm(velocities, axis=1)),
            "avg_size": np.mean(sizes, axis=0),
            "size_std": np.std(sizes, axis=0),
            "history_length": len(self.position_history[tracker_id])
        }


class DetectionStabilizerV2:
    """
    Enhanced detection stabilizer using adaptive size stabilization.
    Wrapper for backward compatibility.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.3,
        use_adaptive_size: bool = True,
        position_smoothing: float = 0.3,
        size_smoothing: float = 0.1
    ):
        self.smoothing_factor = smoothing_factor
        self.use_adaptive_size = use_adaptive_size
        
        if use_adaptive_size:
            self.size_stabilizer = AdaptiveSizeStabilizer(
                history_window=15,
                position_smoothing=position_smoothing,
                size_smoothing_base=size_smoothing
            )
        
        self.history: Dict[int, np.ndarray] = {}
    
    def update(
        self,
        tracker_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """Update and return smoothed bbox."""
        if self.use_adaptive_size:
            return self.size_stabilizer.update(tracker_id, bbox, confidence)
        else:
            return self._simple_smooth(tracker_id, bbox)
    
    def _simple_smooth(
        self,
        tracker_id: int,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Simple EMA smoothing fallback."""
        current = np.array(bbox, dtype=np.float32)
        
        if tracker_id not in self.history:
            self.history[tracker_id] = current
            return tuple(np.round(current).astype(int))
        
        prev = self.history[tracker_id]
        smoothed = self.smoothing_factor * current + (1 - self.smoothing_factor) * prev
        self.history[tracker_id] = smoothed
        
        return tuple(np.round(smoothed).astype(int))
    
    def cleanup_old_trackers(self, active_tracker_ids: set):
        """Remove inactive trackers."""
        if self.use_adaptive_size:
            self.size_stabilizer.cleanup_old_trackers(active_tracker_ids)
        
        to_remove = [tid for tid in self.history if tid not in active_tracker_ids]
        for tid in to_remove:
            del self.history[tid]
    
    def reset(self):
        """Clear all state."""
        if self.use_adaptive_size:
            self.size_stabilizer.position_history.clear()
            self.size_stabilizer.size_history.clear()
            self.size_stabilizer.aspect_ratio_history.clear()
            self.size_stabilizer.velocity_history.clear()
            self.size_stabilizer.smooth_positions.clear()
            self.size_stabilizer.smooth_sizes.clear()
        
        self.history.clear()
