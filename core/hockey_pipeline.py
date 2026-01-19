"""
Hockey Tracking Pipeline with Full Occlusion Handling
======================================================
Production-ready pipeline integrating:
- YOLO detection
- UKF-based tracking
- SAM + CUTIE mask propagation
- Appearance-based re-identification

Usage:
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    
    config = PipelineConfig()
    config.detector.model_path = "path/to/hockey_yolo.pt"
    
    pipeline = HockeyPipeline(config)
    results = pipeline.process_video("input.mp4", "output.mp4")
"""

import numpy as np
import cv2
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from tqdm import tqdm

# Support both installation styles:
# - top-level usage: `from core.pipeline import ...`
# - package usage:   `from hockey_mcbyte_tracker.core...`
try:
    from ..tracking.occlusion_robust_tracker import (
        OcclusionRobustTracker, TrackerConfig, FrameResult
    )
    from ..tracking.enhanced_track import EnhancedTrack
except ImportError:  # pragma: no cover
    from tracking.occlusion_robust_tracker import (
        OcclusionRobustTracker, TrackerConfig, FrameResult
    )
    from tracking.enhanced_track import EnhancedTrack
from .detectors.yolo_detector import UltralyticsYOLODetector
from .utils.visualization import Visualizer, HOCKEY_CLASS_COLORS


@dataclass
class DetectorConfig:
    """YOLO detector configuration."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    device: str = "cuda"
    half_precision: bool = True
    
    # Hockey classes to track
    track_classes: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Center Ice",
        1: "Faceoff",
        2: "Goalpost",
        3: "Goaltender",
        4: "Player",
        5: "Puck",
        6: "Referee"
    })


@dataclass
class MaskConfig:
    """Mask propagation configuration."""
    enabled: bool = True
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_b"
    cutie_checkpoint: Optional[str] = None
    max_internal_size: int = 480


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    show_boxes: bool = True
    show_ids: bool = True
    show_masks: bool = True
    show_trajectories: bool = True
    trajectory_length: int = 30
    show_confidence: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Output settings
    save_mot: bool = False
    save_masks: bool = False


class HockeyPipeline:
    """
    Complete hockey tracking pipeline.
    
    Features:
    - YOLO-based detection with custom model support
    - UKF-based tracking with non-linear motion model
    - SAM + CUTIE mask propagation for identity preservation
    - Appearance-based re-identification
    - Multi-cue association (motion + appearance + mask)
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize detector
        self.detector = UltralyticsYOLODetector(
            model_path=self.config.detector.model_path,
            confidence_threshold=self.config.detector.confidence_threshold,
            nms_threshold=self.config.detector.nms_threshold,
            device=self.config.detector.device,
            half=self.config.detector.half_precision
        )
        
        # Configure tracker
        tracker_config = self.config.tracker
        tracker_config.use_masks = self.config.mask.enabled
        tracker_config.sam_checkpoint = self.config.mask.sam_checkpoint
        tracker_config.cutie_checkpoint = self.config.mask.cutie_checkpoint
        tracker_config.device = self.config.detector.device
        
        # Initialize tracker
        self.tracker = OcclusionRobustTracker(tracker_config)
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            show_boxes=self.config.visualization.show_boxes,
            show_ids=self.config.visualization.show_ids,
            show_masks=self.config.visualization.show_masks,
            show_trajectories=self.config.visualization.show_trajectories,
            trajectory_length=self.config.visualization.trajectory_length,
            class_colors=HOCKEY_CLASS_COLORS,
            class_names=self.config.detector.class_names
        )
        
        # Results storage
        self.mot_results: List[str] = []
        self.frame_times: List[float] = []
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> Tuple[FrameResult, np.ndarray]:
        """
        Process single frame.
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            
        Returns:
            (FrameResult, visualized_frame)
        """
        start_time = time.time()
        
        # Detect
        detections = self.detector.detect(frame)
        
        # Filter by class
        filtered_dets = []
        for det in detections.detections:
            if det.class_id in self.config.detector.track_classes:
                tlwh = np.array([
                    det.bbox[0], det.bbox[1],
                    det.bbox[2] - det.bbox[0],
                    det.bbox[3] - det.bbox[1]
                ])
                filtered_dets.append((tlwh, det.confidence, det.class_id))
        
        # Track
        result = self.tracker.update(filtered_dets, frame)
        
        # Visualize
        vis_frame = self.visualizer.draw_tracks(
            frame, result.tracks,
            mask=result.combined_mask,
            track_to_mask={t.track_id: t.track_id for t in result.tracks}
        )
        
        # Record timing
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        # Record MOT format
        if self.config.save_mot:
            for track in result.tracks:
                tlwh = track.tlwh
                line = (f"{frame_id},{track.track_id},"
                       f"{tlwh[0]:.2f},{tlwh[1]:.2f},"
                       f"{tlwh[2]:.2f},{tlwh[3]:.2f},"
                       f"{track.score:.4f},{track.class_id},-1,-1")
                self.mot_results.append(line)
        
        return result, vis_frame
    
    def process_video(
        self,
        input_path: str,
        output_path: str = None,
        show_progress: bool = True,
        preview: bool = False
    ) -> List[FrameResult]:
        """
        Process complete video.
        
        Args:
            input_path: Input video path
            output_path: Output video path (optional)
            show_progress: Show progress bar
            preview: Show live preview
            
        Returns:
            List of FrameResult for each frame
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[Pipeline] Input: {input_path}")
        print(f"[Pipeline] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
        
        # Setup output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[Pipeline] Output: {output_path}")
        
        # Process frames
        results = []
        self.mot_results = []
        self.frame_times = []
        
        iterator = range(total_frames)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing", unit="frame")
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result, vis_frame = self.process_frame(frame, frame_id)
            results.append(result)
            
            # Write output
            if writer:
                writer.write(vis_frame)
            
            # Preview
            if preview:
                cv2.imshow("Hockey Tracker", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_id += 1
            
            if show_progress:
                iterator.update(1)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        
        # Print statistics
        if self.frame_times:
            avg_fps = 1.0 / np.mean(self.frame_times)
            print(f"\n[Pipeline] Processing complete!")
            print(f"[Pipeline] Average FPS: {avg_fps:.1f}")
            print(f"[Pipeline] Total tracks: {EnhancedTrack._count}")
        
        return results
    
    def save_mot_results(self, output_path: str):
        """Save results in MOT format."""
        with open(output_path, 'w') as f:
            f.write('\n'.join(self.mot_results))
        print(f"[Pipeline] MOT results saved to {output_path}")
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.visualizer.reset_histories()
        self.mot_results = []
        self.frame_times = []


def run_tracking(
    input_path: str,
    output_path: str = None,
    model_path: str = "yolov8n.pt",
    track_classes: List[int] = None,
    use_masks: bool = True,
    preview: bool = False,
    **kwargs
) -> List[FrameResult]:
    """
    Convenience function for quick tracking.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        model_path: YOLO model path
        track_classes: Classes to track
        use_masks: Enable mask propagation
        preview: Show live preview
        **kwargs: Additional config overrides
        
    Returns:
        List of tracking results
    """
    # Create config
    config = PipelineConfig()
    config.detector.model_path = model_path
    config.mask.enabled = use_masks
    
    if track_classes:
        config.detector.track_classes = track_classes
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.detector, key):
            setattr(config.detector, key, value)
        elif hasattr(config.tracker, key):
            setattr(config.tracker, key, value)
    
    # Create and run pipeline
    pipeline = HockeyPipeline(config)
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_tracked.mp4"
    
    results = pipeline.process_video(input_path, output_path, preview=preview)
    
    return results


# Example usage configurations
HOCKEY_CONFIG_BALANCED = """
# Balanced configuration for hockey tracking
# Good balance between speed and accuracy

config = PipelineConfig()

# Detector
config.detector.model_path = "weights/hockey_yolo.pt"
config.detector.confidence_threshold = 0.3
config.detector.track_classes = [3, 4, 5, 6]  # Goalie, Player, Puck, Ref

# Tracker
config.tracker.track_high_thresh = 0.5
config.tracker.track_low_thresh = 0.1
config.tracker.track_buffer = 60  # 2 seconds at 30fps
config.tracker.motion_weight = 0.4
config.tracker.appearance_weight = 0.35
config.tracker.mask_weight = 0.25

# Masks
config.mask.enabled = True
config.mask.sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
"""

HOCKEY_CONFIG_FAST = """
# Fast configuration for real-time tracking
# Prioritizes speed over accuracy

config = PipelineConfig()

# Detector
config.detector.model_path = "weights/hockey_yolo.pt"
config.detector.confidence_threshold = 0.4
config.detector.half_precision = True

# Tracker - simpler matching
config.tracker.use_masks = False  # Disable for speed
config.tracker.use_reid = False
config.tracker.motion_weight = 0.7
config.tracker.appearance_weight = 0.3

# No masks
config.mask.enabled = False
"""

HOCKEY_CONFIG_ACCURATE = """
# High accuracy configuration
# Best identity preservation, slower processing

config = PipelineConfig()

# Detector
config.detector.model_path = "weights/hockey_yolo.pt"
config.detector.confidence_threshold = 0.2  # Lower threshold
config.detector.nms_threshold = 0.5

# Tracker - full features
config.tracker.track_buffer = 90  # 3 seconds
config.tracker.confirm_frames = 2
config.tracker.motion_weight = 0.3
config.tracker.appearance_weight = 0.35
config.tracker.mask_weight = 0.35

# Full mask support
config.mask.enabled = True
config.mask.sam_checkpoint = "weights/sam_vit_h_4b8939.pth"  # Larger SAM
config.mask.cutie_checkpoint = "weights/cutie-base-mega.pth"
"""
