"""
Detectors Module
================
Object detection components.
"""

from .yolo_detector import (
    UltralyticsYOLODetector,
    Detection,
    DetectionResult
)

__all__ = [
    'UltralyticsYOLODetector',
    'Detection',
    'DetectionResult',
]
