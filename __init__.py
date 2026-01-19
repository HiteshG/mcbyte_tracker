"""
Hockey McByte Tracker
=====================
Production-ready multi-object tracking for hockey with occlusion handling.

Features:
- Unscented Kalman Filter for non-linear state estimation
- SAM + CUTIE mask propagation for identity preservation
- Appearance-based re-identification
- Multi-cue association (motion + appearance + mask)

Quick Start:
    from hockey_mcbyte_tracker import HockeyPipeline, PipelineConfig
    
    config = PipelineConfig()
    config.detector.model_path = "path/to/hockey_yolo.pt"
    config.detector.track_classes = [3, 4, 5, 6]  # Goalie, Player, Puck, Ref
    
    pipeline = HockeyPipeline(config)
    results = pipeline.process_video("input.mp4", "output.mp4")
"""

__version__ = "2.0.0"
__author__ = "Hockey Tracking System"

# Main pipeline
from .core.hockey_pipeline import (
    HockeyPipeline,
    PipelineConfig,
    DetectorConfig,
    MaskConfig,
    VisualizationConfig,
    run_tracking
)

# Tracker components
from .tracking.occlusion_robust_tracker import (
    OcclusionRobustTracker,
    TrackerConfig,
    FrameResult
)
from .tracking.enhanced_track import EnhancedTrack, TrackState

# Mask propagation
from .mask_propagation.integrated_masks import (
    MaskPropagationSystem,
    MaskIdentityVerifier
)

# Detection
from .detectors.yolo_detector import UltralyticsYOLODetector

# Visualization
from .utils.visualization import Visualizer, HOCKEY_CLASS_COLORS

__all__ = [
    # Version
    '__version__',
    
    # Pipeline
    'HockeyPipeline',
    'PipelineConfig',
    'DetectorConfig',
    'MaskConfig',
    'VisualizationConfig',
    'run_tracking',
    
    # Tracker
    'OcclusionRobustTracker',
    'TrackerConfig',
    'FrameResult',
    'EnhancedTrack',
    'TrackState',
    
    # Masks
    'MaskPropagationSystem',
    'MaskIdentityVerifier',
    
    # Detection
    'UltralyticsYOLODetector',
    
    # Visualization
    'Visualizer',
    'HOCKEY_CLASS_COLORS',
]
