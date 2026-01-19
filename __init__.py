"""
Hockey McByte Tracker
=====================
Production-ready multi-object tracking for ice hockey using
YOLO detection with McByte tracking and Cutie mask propagation.
"""

__version__ = "1.0.0"
__author__ = "Hockey Analytics"

from configs.config import PipelineConfig, DetectorConfig, TrackerConfig, MaskConfig
from core.pipeline import HockeyTrackingPipeline, run_tracking
from detectors.yolo_detector import UltralyticsYOLODetector, create_detector
from tracking.mcbyte_tracker import McByteTracker
from tracking.track import Track, TrackState

__all__ = [
    "PipelineConfig",
    "DetectorConfig", 
    "TrackerConfig",
    "MaskConfig",
    "HockeyTrackingPipeline",
    "run_tracking",
    "UltralyticsYOLODetector",
    "create_detector",
    "McByteTracker",
    "Track",
    "TrackState",
]
