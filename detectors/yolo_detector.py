"""
Custom YOLO Detector Wrapper
============================
Production-ready wrapper for custom YOLO detection models.
Supports multiple YOLO implementations (Ultralytics, YOLOX, etc.)
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2


@dataclass
class Detection:
    """Single detection result."""
    bbox: np.ndarray        # [x1, y1, x2, y2] format
    confidence: float
    class_id: int
    class_name: str = ""
    
    @property
    def tlwh(self) -> np.ndarray:
        """Convert to [top, left, width, height] format."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])
    
    @property
    def tlbr(self) -> np.ndarray:
        """Return [top, left, bottom, right] format."""
        return self.bbox.copy()
    
    @property
    def center(self) -> np.ndarray:
        """Return center point [cx, cy]."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    @property
    def area(self) -> float:
        """Return bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    detections: List[Detection]
    frame_id: int
    image_size: Tuple[int, int]  # (height, width)
    
    def filter_by_class(self, class_ids: List[int]) -> 'FrameDetections':
        """Filter detections by class ID."""
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return FrameDetections(filtered, self.frame_id, self.image_size)
    
    def filter_by_confidence(self, min_conf: float) -> 'FrameDetections':
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_conf]
        return FrameDetections(filtered, self.frame_id, self.image_size)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [N, 6]: [x1, y1, x2, y2, conf, class_id]."""
        if not self.detections:
            return np.empty((0, 6))
        return np.array([
            [*d.bbox, d.confidence, d.class_id] 
            for d in self.detections
        ])
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __iter__(self):
        return iter(self.detections)


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> FrameDetections:
        """Run detection on a single image."""
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """Warm up the model with dummy inference."""
        pass


class UltralyticsYOLODetector(BaseDetector):
    """
    Wrapper for Ultralytics YOLO models (YOLOv5, YOLOv8, YOLO11, etc.)
    
    This is the recommended detector for custom trained models.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        track_classes: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        device: str = "cuda:0",
        half_precision: bool = True,
        input_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO weights (.pt file)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            track_classes: List of class IDs to track (None = all)
            class_names: Mapping of class ID to name
            device: Torch device string
            half_precision: Use FP16 inference
            input_size: Input image size (height, width)
        """
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold
        self.track_classes = track_classes
        self.class_names = class_names or {}
        self.device = device
        self.half = half_precision
        self.input_size = input_size
        self.frame_id = 0
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            # Set device
            if 'cuda' in self.device and torch.cuda.is_available():
                self.model.to(self.device)
                if self.half:
                    self.model.model.half()
            else:
                self.device = 'cpu'
                self.model.to('cpu')
                self.half = False
                
            print(f"[Detector] Loaded YOLO model from {self.model_path}")
            print(f"[Detector] Device: {self.device}, FP16: {self.half}")
            
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
    
    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        _ = self.detect(dummy)
        print("[Detector] Warmup complete")
    
    def detect(self, image: np.ndarray) -> FrameDetections:
        """
        Run detection on image.
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            FrameDetections object with all detections
        """
        self.frame_id += 1
        h, w = image.shape[:2]
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            verbose=False,
            device=self.device,
        )[0]
        
        # Parse results
        detections = []
        if results.boxes is not None and len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for bbox, conf, cls_id in zip(boxes, confs, classes):
                # Filter by tracked classes if specified
                if self.track_classes and cls_id not in self.track_classes:
                    continue
                    
                detections.append(Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}")
                ))
        
        return FrameDetections(
            detections=detections,
            frame_id=self.frame_id,
            image_size=(h, w)
        )


class DetectionFileReader:
    """
    Read detections from file (MOT format or custom format).
    
    Useful for using pre-computed detections or oracle detections.
    """
    
    def __init__(
        self,
        det_file: str,
        track_classes: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
        min_confidence: float = 0.0,
    ):
        """
        Initialize detection file reader.
        
        Args:
            det_file: Path to detection file
            track_classes: List of class IDs to track
            class_names: Mapping of class ID to name
            min_confidence: Minimum confidence threshold
        """
        self.det_file = det_file
        self.track_classes = track_classes
        self.class_names = class_names or {}
        self.min_conf = min_confidence
        
        self.detections_by_frame = self._load_detections()
    
    def _load_detections(self) -> Dict[int, List[Detection]]:
        """Load detections from file."""
        detections = {}
        
        with open(self.det_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                # MOT format: frame, id, x, y, w, h, conf, class_id, ...
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                cls_id = int(parts[7]) if len(parts) > 7 else 0
                
                # Filter
                if conf < self.min_conf:
                    continue
                if self.track_classes and cls_id not in self.track_classes:
                    continue
                
                if frame_id not in detections:
                    detections[frame_id] = []
                    
                detections[frame_id].append(Detection(
                    bbox=np.array([x, y, x + w, y + h]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}")
                ))
        
        return detections
    
    def get_detections(self, frame_id: int, image_size: Tuple[int, int]) -> FrameDetections:
        """Get detections for a specific frame."""
        dets = self.detections_by_frame.get(frame_id, [])
        return FrameDetections(dets, frame_id, image_size)


def create_detector(config) -> BaseDetector:
    """
    Factory function to create appropriate detector.
    
    Args:
        config: DetectorConfig object
        
    Returns:
        Configured detector instance
    """
    return UltralyticsYOLODetector(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        track_classes=config.track_classes,
        class_names=config.class_names,
        device=config.device,
        half_precision=config.half_precision,
        input_size=config.input_size,
    )
