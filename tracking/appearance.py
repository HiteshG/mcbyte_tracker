"""
Appearance Feature Extractor for Re-Identification
===================================================
Extracts and manages appearance features for identity preservation.

Features:
- Color histogram features (fast, robust)
- Deep CNN features (optional, more accurate)
- Feature gallery with temporal smoothing
- Occlusion-aware feature updates
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class AppearanceFeature:
    """Container for appearance features."""
    histogram: np.ndarray          # Color histogram
    deep_feature: Optional[np.ndarray] = None  # CNN feature
    mask_feature: Optional[np.ndarray] = None  # Mask-based feature
    quality: float = 1.0           # Feature quality score
    frame_id: int = 0              # When extracted


@dataclass 
class FeatureGallery:
    """
    Feature gallery for a single track.
    Maintains multiple features for robust matching.
    """
    track_id: int
    features: deque = field(default_factory=lambda: deque(maxlen=30))
    smooth_feature: Optional[np.ndarray] = None
    smooth_histogram: Optional[np.ndarray] = None
    last_update: int = 0
    update_count: int = 0
    
    def add_feature(self, feature: AppearanceFeature, frame_id: int):
        """Add new feature to gallery."""
        self.features.append(feature)
        self.last_update = frame_id
        self.update_count += 1
        
        # Update smoothed features
        self._update_smooth_features()
    
    def _update_smooth_features(self):
        """Compute smoothed feature from gallery."""
        if len(self.features) == 0:
            return
        
        # Weight by quality and recency
        weights = []
        histograms = []
        deep_feats = []
        
        for i, feat in enumerate(self.features):
            # More recent = higher weight
            recency_weight = (i + 1) / len(self.features)
            weight = feat.quality * recency_weight
            weights.append(weight)
            histograms.append(feat.histogram)
            
            if feat.deep_feature is not None:
                deep_feats.append(feat.deep_feature)
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Weighted average of histograms
        self.smooth_histogram = np.average(histograms, axis=0, weights=weights)
        self.smooth_histogram /= (self.smooth_histogram.sum() + 1e-6)
        
        # Weighted average of deep features
        if deep_feats:
            self.smooth_feature = np.average(deep_feats, axis=0, weights=weights[-len(deep_feats):])
            self.smooth_feature /= (np.linalg.norm(self.smooth_feature) + 1e-6)


class AppearanceExtractor:
    """
    Extracts appearance features from image crops.
    
    Supports multiple feature types:
    - Color histograms (H, S channels in HSV)
    - Deep CNN features (optional)
    - Mask-weighted features
    """
    
    def __init__(
        self,
        use_deep_features: bool = True,
        deep_model_path: Optional[str] = None,
        histogram_bins: Tuple[int, int] = (16, 16),
        feature_dim: int = 512,
        device: str = "cuda"
    ):
        """
        Initialize appearance extractor.
        
        Args:
            use_deep_features: Whether to use CNN features
            deep_model_path: Path to ReID model (e.g., OSNet)
            histogram_bins: Bins for H, S histograms
            feature_dim: Dimension of deep features
            device: Device for CNN inference
        """
        self.use_deep_features = use_deep_features
        self.histogram_bins = histogram_bins
        self.feature_dim = feature_dim
        self.device = device
        
        # Feature galleries per track
        self.galleries: Dict[int, FeatureGallery] = {}
        
        # Deep model (lazy loaded)
        self._deep_model = None
        self._deep_model_path = deep_model_path
        self._model_loaded = False
        
        # Preprocessing
        self.crop_size = (128, 256)  # W, H for ReID models
    
    def _load_deep_model(self):
        """Lazy load deep ReID model."""
        if self._model_loaded:
            return
        
        if not self.use_deep_features:
            self._model_loaded = True
            return
        
        try:
            import torch
            import torchvision.transforms as T
            
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size[::-1]),  # H, W
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
            
            # Try to load OSNet or similar
            if self._deep_model_path:
                # Load custom model
                self._deep_model = torch.jit.load(self._deep_model_path)
            else:
                # Try boxmot's OSNet
                try:
                    from boxmot.appearance.reid_model_factory import (
                        load_pretrained_weights, get_model_name, build_model
                    )
                    model_name = "osnet_x0_25"
                    self._deep_model = build_model(model_name, 1, loss='softmax')
                    # Will use random weights if no pretrained available
                except ImportError:
                    # Fallback: simple CNN feature extractor
                    self._deep_model = self._build_simple_cnn()
            
            if self._deep_model is not None:
                self._deep_model.eval()
                if torch.cuda.is_available() and self.device == "cuda":
                    self._deep_model = self._deep_model.cuda()
            
            self._model_loaded = True
            
        except ImportError:
            print("[AppearanceExtractor] PyTorch not available, using histogram features only")
            self.use_deep_features = False
            self._model_loaded = True
    
    def _build_simple_cnn(self):
        """Build simple CNN for feature extraction."""
        import torch
        import torch.nn as nn
        
        class SimpleCNN(nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 2)),
                )
                self.fc = nn.Linear(128 * 4 * 2, feature_dim)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleCNN(self.feature_dim)
    
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        mask: Optional[np.ndarray] = None,
        frame_id: int = 0
    ) -> AppearanceFeature:
        """
        Extract appearance features from bounding box region.
        
        Args:
            frame: Full frame (BGR)
            bbox: [x1, y1, x2, y2] bounding box
            mask: Optional binary mask for the object
            frame_id: Current frame number
            
        Returns:
            AppearanceFeature containing all features
        """
        # Extract crop
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return empty feature
            return AppearanceFeature(
                histogram=np.zeros(sum(self.histogram_bins)),
                quality=0.0,
                frame_id=frame_id
            )
        
        crop = frame[y1:y2, x1:x2]
        
        # Extract crop mask if available
        crop_mask = None
        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2]
        
        # Compute quality score
        quality = self._compute_quality(crop, crop_mask)
        
        # Extract histogram features
        histogram = self._extract_histogram(crop, crop_mask)
        
        # Extract deep features
        deep_feature = None
        if self.use_deep_features:
            self._load_deep_model()
            if self._deep_model is not None:
                deep_feature = self._extract_deep_feature(crop)
        
        return AppearanceFeature(
            histogram=histogram,
            deep_feature=deep_feature,
            quality=quality,
            frame_id=frame_id
        )
    
    def _extract_histogram(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract color histogram from crop.
        
        Uses HSV color space for illumination robustness.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Prepare mask
        cv_mask = None
        if mask is not None and mask.size > 0:
            cv_mask = (mask > 0).astype(np.uint8) * 255
        
        # Compute H and S histograms
        h_hist = cv2.calcHist(
            [hsv], [0], cv_mask, 
            [self.histogram_bins[0]], [0, 180]
        ).flatten()
        
        s_hist = cv2.calcHist(
            [hsv], [1], cv_mask,
            [self.histogram_bins[1]], [0, 256]
        ).flatten()
        
        # Concatenate and normalize
        histogram = np.concatenate([h_hist, s_hist])
        histogram = histogram / (histogram.sum() + 1e-6)
        
        return histogram
    
    def _extract_deep_feature(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract deep CNN features."""
        try:
            import torch
            
            # Resize and transform
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0)
            
            if torch.cuda.is_available() and self.device == "cuda":
                tensor = tensor.cuda()
            
            # Extract features
            with torch.no_grad():
                features = self._deep_model(tensor)
            
            features = features.cpu().numpy().flatten()
            features = features / (np.linalg.norm(features) + 1e-6)
            
            return features
            
        except Exception as e:
            return None
    
    def _compute_quality(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute feature quality score.
        
        Based on:
        - Crop size (larger = better)
        - Blur level (sharper = better)
        - Mask coverage (more visible = better)
        """
        h, w = crop.shape[:2]
        
        # Size score
        size_score = min(1.0, (w * h) / (64 * 128))
        
        # Blur score (Laplacian variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500)
        
        # Mask coverage score
        mask_score = 1.0
        if mask is not None and mask.size > 0:
            mask_score = mask.sum() / mask.size
        
        # Combined quality
        quality = 0.3 * size_score + 0.4 * blur_score + 0.3 * mask_score
        
        return float(quality)
    
    def update_gallery(
        self,
        track_id: int,
        feature: AppearanceFeature,
        frame_id: int,
        min_quality: float = 0.3
    ):
        """
        Update feature gallery for a track.
        
        Args:
            track_id: Track ID
            feature: Extracted feature
            frame_id: Current frame
            min_quality: Minimum quality to add to gallery
        """
        if feature.quality < min_quality:
            return
        
        if track_id not in self.galleries:
            self.galleries[track_id] = FeatureGallery(track_id=track_id)
        
        self.galleries[track_id].add_feature(feature, frame_id)
    
    def get_gallery(self, track_id: int) -> Optional[FeatureGallery]:
        """Get feature gallery for track."""
        return self.galleries.get(track_id)
    
    def compute_distance(
        self,
        feature1: AppearanceFeature,
        feature2: AppearanceFeature,
        use_deep: bool = True
    ) -> float:
        """
        Compute appearance distance between two features.
        
        Args:
            feature1: First feature
            feature2: Second feature
            use_deep: Whether to use deep features
            
        Returns:
            Distance (0 = identical, 1 = completely different)
        """
        # Histogram distance (Bhattacharyya)
        hist_dist = cv2.compareHist(
            feature1.histogram.astype(np.float32),
            feature2.histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        # Deep feature distance (cosine)
        deep_dist = 1.0
        if (use_deep and 
            feature1.deep_feature is not None and 
            feature2.deep_feature is not None):
            cosine_sim = np.dot(feature1.deep_feature, feature2.deep_feature)
            deep_dist = 1.0 - cosine_sim
        
        # Combine distances
        if feature1.deep_feature is not None and feature2.deep_feature is not None:
            return 0.3 * hist_dist + 0.7 * deep_dist
        else:
            return hist_dist
    
    def compute_gallery_distance(
        self,
        feature: AppearanceFeature,
        gallery: FeatureGallery
    ) -> float:
        """
        Compute distance between feature and gallery.
        
        Uses smoothed gallery feature for robustness.
        """
        if gallery.smooth_histogram is None:
            return 1.0
        
        # Histogram distance
        hist_dist = cv2.compareHist(
            feature.histogram.astype(np.float32),
            gallery.smooth_histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        # Deep feature distance
        deep_dist = 1.0
        if (feature.deep_feature is not None and 
            gallery.smooth_feature is not None):
            cosine_sim = np.dot(feature.deep_feature, gallery.smooth_feature)
            deep_dist = 1.0 - cosine_sim
            return 0.3 * hist_dist + 0.7 * deep_dist
        
        return hist_dist
    
    def cleanup(self, active_track_ids: set):
        """Remove galleries for inactive tracks."""
        to_remove = [tid for tid in self.galleries if tid not in active_track_ids]
        for tid in to_remove:
            del self.galleries[tid]


def compute_appearance_cost_matrix(
    extractor: AppearanceExtractor,
    track_features: Dict[int, AppearanceFeature],
    detection_features: List[AppearanceFeature],
    track_ids: List[int]
) -> np.ndarray:
    """
    Compute appearance cost matrix for assignment.
    
    Args:
        extractor: AppearanceExtractor instance
        track_features: Features for each track (or use gallery)
        detection_features: Features for each detection
        track_ids: List of track IDs (in order)
        
    Returns:
        Cost matrix (num_tracks, num_detections)
    """
    num_tracks = len(track_ids)
    num_dets = len(detection_features)
    
    if num_tracks == 0 or num_dets == 0:
        return np.zeros((num_tracks, num_dets))
    
    cost_matrix = np.zeros((num_tracks, num_dets))
    
    for i, tid in enumerate(track_ids):
        gallery = extractor.get_gallery(tid)
        
        for j, det_feat in enumerate(detection_features):
            if gallery is not None:
                cost_matrix[i, j] = extractor.compute_gallery_distance(
                    det_feat, gallery
                )
            elif tid in track_features:
                cost_matrix[i, j] = extractor.compute_distance(
                    track_features[tid], det_feat
                )
            else:
                cost_matrix[i, j] = 1.0
    
    return cost_matrix
