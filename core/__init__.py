"""
Core Pipeline Module
====================
Main pipeline components for hockey tracking.
"""

from .hockey_pipeline import (
    HockeyPipeline,
    PipelineConfig,
    DetectorConfig,
    MaskConfig,
    VisualizationConfig,
    run_tracking
)

# Legacy
from .pipeline import HockeyTrackingPipeline, VideoReader, VideoWriter

__all__ = [
    'HockeyPipeline',
    'PipelineConfig',
    'DetectorConfig',
    'MaskConfig',
    'VisualizationConfig',
    'run_tracking',
    
    # Legacy
    'HockeyTrackingPipeline',
    'VideoReader',
    'VideoWriter',
]
