"""
Tracking Module
===============
Contains all tracking-related components.
"""

from .enhanced_track import (
    EnhancedTrack,
    TrackState,
    TrackMemory,
    iou_batch,
    mask_iou_batch,
    linear_assignment,
    fuse_motion_appearance
)
from .ukf import UnscentedKalmanFilter, AdaptiveUKF
from .appearance import AppearanceExtractor, AppearanceFeature, FeatureGallery
from .occlusion_robust_tracker import (
    OcclusionRobustTracker,
    TrackerConfig,
    FrameResult,
    CameraMotionCompensator
)

# Legacy imports for backward compatibility
from .kalman_filter import KalmanFilter
from .track import Track
from .mcbyte_tracker import McByteTracker

__all__ = [
    # New robust tracker
    'OcclusionRobustTracker',
    'TrackerConfig', 
    'FrameResult',
    'EnhancedTrack',
    'TrackState',
    'TrackMemory',
    
    # UKF
    'UnscentedKalmanFilter',
    'AdaptiveUKF',
    
    # Appearance
    'AppearanceExtractor',
    'AppearanceFeature',
    'FeatureGallery',
    
    # Utilities
    'CameraMotionCompensator',
    'iou_batch',
    'mask_iou_batch',
    'linear_assignment',
    
    # Legacy
    'KalmanFilter',
    'Track',
    'McByteTracker',
]
