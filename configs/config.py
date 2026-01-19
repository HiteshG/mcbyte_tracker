"""
Hockey McByte Tracker Configuration
===================================
Centralized configuration management for the tracking pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TrackableClass(Enum):
    """Hockey-specific detection classes from custom YOLO model."""
    CENTER_ICE = 0
    FACEOFF = 1
    GOALPOST = 2
    GOALTENDER = 3
    PLAYER = 4
    PUCK = 5
    REFEREE = 6


@dataclass
class DetectorConfig:
    """Configuration for YOLO detector."""
    model_path: str = "weights/hockey_yolo.pt"
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    input_size: Tuple[int, int] = (1280, 720)
    device: str = "cuda:0"
    half_precision: bool = True
    
    # Classes to track (subset of all detected classes)
    track_classes: List[int] = field(default_factory=lambda: [
        TrackableClass.PLAYER.value,
        TrackableClass.PUCK.value,
        TrackableClass.REFEREE.value,
        TrackableClass.GOALTENDER.value,
    ])
    
    # Class names mapping
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
class TrackerConfig:
    """Configuration for McByte tracker."""
    # Detection thresholds
    track_thresh: float = 0.5      # High confidence detection threshold
    track_buffer: int = 30         # Frames to keep lost tracks
    match_thresh: float = 0.8      # IoU threshold for matching
    
    # Second association (low score detections)
    low_thresh: float = 0.1        # Low confidence threshold
    new_track_thresh: float = 0.6  # Threshold for initializing new tracks
    
    # Kalman filter settings
    use_kalman: bool = True
    
    # Camera motion compensation
    use_gmc: bool = True
    gmc_method: str = "sparseOptFlow"  # Options: sparseOptFlow, orb, ecc
    
    # Frame rate for motion prediction
    frame_rate: int = 30


@dataclass
class MaskConfig:
    """Configuration for mask propagation (Cutie + SAM)."""
    # Cutie settings
    cutie_weights: str = "weights/cutie-base-mega.pth"
    max_internal_size: int = 480   # -1 for original size
    mem_every: int = 5             # Memory update frequency
    use_long_term: bool = True
    
    # SAM settings  
    sam_weights: str = "weights/sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    
    # Mask creation settings
    mask_start_frame: int = 1
    bbox_overlap_threshold: float = 0.6
    
    # Device
    device: str = "cuda:0"


@dataclass
class VisualizationConfig:
    """Configuration for output visualization."""
    show_boxes: bool = True
    show_ids: bool = True
    show_masks: bool = True
    show_trajectories: bool = True
    trajectory_length: int = 30
    
    # Colors per class (BGR format)
    class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        3: (255, 0, 0),      # Goaltender - Blue
        4: (0, 255, 0),      # Player - Green
        5: (0, 255, 255),    # Puck - Yellow
        6: (0, 0, 255),      # Referee - Red
    })
    
    # Output settings
    save_video: bool = True
    save_frames: bool = False
    output_fps: int = 30
    codec: str = "mp4v"


@dataclass
class PipelineConfig:
    """Master configuration combining all components."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # I/O settings
    input_path: str = ""
    output_dir: str = "outputs"
    
    # Processing settings
    start_frame: int = 0
    end_frame: int = -1  # -1 for all frames
    
    # Logging
    verbose: bool = True
    save_logs: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def get_default_config() -> PipelineConfig:
    """Get default configuration for hockey tracking."""
    return PipelineConfig()
