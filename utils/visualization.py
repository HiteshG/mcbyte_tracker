"""
Visualization Utilities
=======================
Drawing functions for tracking results visualization.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# Color palette for visualization (BGR format)
PALETTE = [
    (255, 128, 0),    # 0: Orange
    (0, 255, 0),      # 1: Green
    (0, 0, 255),      # 2: Red
    (255, 0, 255),    # 3: Magenta
    (0, 255, 255),    # 4: Yellow
    (255, 255, 0),    # 5: Cyan
    (128, 0, 255),    # 6: Purple
    (0, 128, 255),    # 7: Light Orange
    (255, 128, 128),  # 8: Light Blue
    (128, 255, 128),  # 9: Light Green
    (128, 128, 255),  # 10: Light Red
    (255, 255, 128),  # 11: Light Cyan
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """Get color for track ID."""
    return PALETTE[idx % len(PALETTE)]


class Visualizer:
    """
    Tracking visualization manager.
    
    Handles drawing of bounding boxes, IDs, trajectories, and masks.
    """
    
    def __init__(
        self,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_masks: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 30,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize visualizer.
        
        Args:
            show_boxes: Draw bounding boxes
            show_ids: Draw track IDs
            show_masks: Draw segmentation masks
            show_trajectories: Draw track history
            trajectory_length: Max trajectory points
            class_colors: Custom colors per class (BGR)
            class_names: Class ID to name mapping
        """
        self.show_boxes = show_boxes
        self.show_ids = show_ids
        self.show_masks = show_masks
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.class_colors = class_colors or {}
        self.class_names = class_names or {}
        
        # Track history for trajectories
        self.track_histories: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List,
        mask: Optional[np.ndarray] = None,
        track_to_mask: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Draw all tracking visualizations on frame.
        
        Args:
            frame: Input frame (BGR)
            tracks: List of Track objects
            mask: Optional segmentation mask
            track_to_mask: Track ID to mask ID mapping
            
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        # Draw masks first (behind boxes)
        if self.show_masks and mask is not None and track_to_mask is not None:
            vis = self._draw_masks(vis, mask, tracks, track_to_mask)
        
        # Draw trajectories
        if self.show_trajectories:
            vis = self._draw_trajectories(vis, tracks)
        
        # Draw boxes and IDs
        for track in tracks:
            color = self._get_track_color(track)
            
            if self.show_boxes:
                vis = self._draw_box(vis, track, color)
            
            if self.show_ids:
                vis = self._draw_id(vis, track, color)
        
        return vis
    
    def _get_track_color(self, track) -> Tuple[int, int, int]:
        """Get color for track based on class or ID."""
        if hasattr(track, 'class_id') and track.class_id in self.class_colors:
            return self.class_colors[track.class_id]
        return get_color(track.track_id)
    
    def _draw_box(
        self,
        frame: np.ndarray,
        track,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw bounding box."""
        tlbr = track.tlbr.astype(int)
        x1, y1, x2, y2 = tlbr
        
        # Draw box with thickness based on confidence
        thickness = 2
        if hasattr(track, 'score'):
            thickness = max(1, int(track.score * 4))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        return frame
    
    def _draw_id(
        self,
        frame: np.ndarray,
        track,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw track ID label."""
        tlbr = track.tlbr.astype(int)
        x1, y1 = tlbr[:2]
        
        # Build label
        label = f"ID:{track.track_id}"
        if hasattr(track, 'class_id'):
            class_name = self.class_names.get(track.class_id, "")
            if class_name:
                label = f"{class_name} {label}"
        
        if hasattr(track, 'score'):
            label += f" {track.score:.2f}"
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        y1_label = max(y1 - h - 10, 0)
        cv2.rectangle(frame, (x1, y1_label), (x1 + w, y1_label + h + 8), color, -1)
        
        # Draw text
        cv2.putText(
            frame, label, (x1, y1_label + h + 4),
            font, font_scale, (255, 255, 255), thickness
        )
        
        return frame
    
    def _draw_trajectories(
        self,
        frame: np.ndarray,
        tracks: List
    ) -> np.ndarray:
        """Draw track trajectories."""
        for track in tracks:
            # Update history
            center = track.xywh[:2].astype(int)
            self.track_histories[track.track_id].append(tuple(center))
            
            # Trim history
            if len(self.track_histories[track.track_id]) > self.trajectory_length:
                self.track_histories[track.track_id] = \
                    self.track_histories[track.track_id][-self.trajectory_length:]
            
            # Draw trajectory
            history = self.track_histories[track.track_id]
            if len(history) < 2:
                continue
            
            color = self._get_track_color(track)
            pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, 2)
        
        return frame
    
    def _draw_masks(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        tracks: List,
        track_to_mask: Dict[int, int]
    ) -> np.ndarray:
        """Draw segmentation masks with transparency."""
        overlay = frame.copy()
        alpha = 0.4
        
        for track in tracks:
            if track.track_id not in track_to_mask:
                continue
            
            mask_id = track_to_mask[track.track_id]
            track_mask = mask == mask_id
            
            if track_mask.sum() == 0:
                continue
            
            color = self._get_track_color(track)
            overlay[track_mask] = color
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def reset_histories(self) -> None:
        """Clear all trajectory histories."""
        self.track_histories.clear()
    
    def remove_track(self, track_id: int) -> None:
        """Remove trajectory history for specific track."""
        if track_id in self.track_histories:
            del self.track_histories[track_id]


def draw_detections(
    frame: np.ndarray,
    detections,
    class_names: Optional[Dict[int, str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw detection bounding boxes (before tracking).
    
    Args:
        frame: Input frame (BGR)
        detections: FrameDetections or list of Detection objects
        class_names: Class ID to name mapping
        color: Box color (BGR)
        
    Returns:
        Annotated frame
    """
    vis = frame.copy()
    
    if hasattr(detections, 'detections'):
        det_list = detections.detections
    else:
        det_list = detections
    
    for det in det_list:
        if hasattr(det, 'bbox'):
            x1, y1, x2, y2 = det.bbox.astype(int)
            conf = det.confidence
            cls_id = det.class_id
        else:
            tlwh, conf = det[:2]
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
            cls_id = det[2] if len(det) > 2 else 0
        
        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{conf:.2f}"
        if class_names and cls_id in class_names:
            label = f"{class_names[cls_id]}: {label}"
        
        cv2.putText(
            vis, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
    return vis


def create_info_panel(
    width: int,
    height: int,
    info: Dict[str, any]
) -> np.ndarray:
    """
    Create information panel with tracking statistics.
    
    Args:
        width: Panel width
        height: Panel height
        info: Dictionary of info to display
        
    Returns:
        Info panel image
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(
            panel, text, (10, y_offset),
            font, 0.6, (255, 255, 255), 1
        )
        y_offset += 25
    
    return panel


def save_tracking_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = "mp4v"
) -> None:
    """
    Save list of frames as video.
    
    Args:
        frames: List of BGR frames
        output_path: Output video path
        fps: Frame rate
        codec: Video codec
    """
    if len(frames) == 0:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    print(f"[Visualizer] Saved video to {output_path}")
