"""
Mask Propagation Manager
========================
Optimized mask propagation using Cutie (temporal propagation) 
and SAM (initial mask generation) for track-aware segmentation.

Architecture:
1. SAM generates initial masks from bounding boxes
2. Cutie propagates masks temporally
3. Masks provide association cues for tracking
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import cv2


@dataclass
class MaskState:
    """Current state of mask propagation."""
    prediction: Optional[np.ndarray] = None
    track_to_mask: Dict[int, int] = None
    mask_colors: Dict[int, int] = None
    confidence: Dict[int, float] = None
    
    def __post_init__(self):
        if self.track_to_mask is None:
            self.track_to_mask = {}
        if self.mask_colors is None:
            self.mask_colors = {}
        if self.confidence is None:
            self.confidence = {}


class MaskPropagator:
    """
    Mask propagation manager using Cutie for temporal propagation.
    
    Provides segmentation masks that persist across frames for
    improved tracking association.
    """
    
    def __init__(
        self,
        cutie_weights: str = "weights/cutie-base-mega.pth",
        sam_weights: str = "weights/sam_vit_b_01ec64.pth",
        sam_model_type: str = "vit_b",
        device: str = "cuda:0",
        max_internal_size: int = 480,
        mem_every: int = 5,
    ):
        """
        Initialize mask propagation components.
        
        Args:
            cutie_weights: Path to Cutie model weights
            sam_weights: Path to SAM model weights
            sam_model_type: SAM model variant
            device: Torch device
            max_internal_size: Internal processing resolution
            mem_every: Memory update frequency for Cutie
        """
        self.device = device
        self.max_internal_size = max_internal_size
        self.mem_every = mem_every
        self.cutie_weights = cutie_weights
        self.sam_weights = sam_weights
        self.sam_model_type = sam_model_type
        
        # State
        self.state = MaskState()
        self.frame_count = 0
        self.initialized = False
        self.mask_id_counter = 0
        self.awaiting_masks: List[int] = []  # Track IDs waiting for mask creation
        
        # Lazy initialization
        self._sam = None
        self._sam_predictor = None
        self._cutie = None
        self._processor = None
    
    def _init_sam(self) -> None:
        """Initialize SAM model (lazy loading)."""
        if self._sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            self._sam = sam_model_registry[self.sam_model_type](
                checkpoint=self.sam_weights
            )
            self._sam.to(device=self.device)
            self._sam_predictor = SamPredictor(self._sam)
            
            print(f"[MaskPropagator] SAM initialized: {self.sam_model_type}")
            
        except ImportError:
            raise ImportError(
                "segment_anything not found. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
    
    def _init_cutie(self) -> None:
        """Initialize Cutie model (lazy loading)."""
        if self._cutie is not None:
            return
        
        try:
            import sys
            from pathlib import Path
            
            # Add Cutie to path (adjust based on your installation)
            # You may need to modify this based on where Cutie is installed
            
            from omegaconf import OmegaConf
            from cutie.model.cutie import CUTIE
            from cutie.inference.inference_core import InferenceCore
            
            # Create config
            cfg = OmegaConf.create({
                'model': {
                    'type': 'base',
                },
                'weights': self.cutie_weights,
                'max_internal_size': self.max_internal_size,
                'mem_every': self.mem_every,
                'use_long_term': True,
                'long_term': {
                    'count_usage': True,
                    'max_mem_frames': 10,
                    'min_mem_frames': 5,
                    'num_prototypes': 128,
                    'max_num_tokens': 10000,
                    'buffer_tokens': 2000,
                },
                'top_k': 30,
                'amp': True,
            })
            
            # Load model
            self._cutie = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(self.cutie_weights, map_location=self.device)
            self._cutie.load_weights(model_weights)
            
            # Create processor
            self._processor = InferenceCore(self._cutie, cfg=cfg)
            
            print(f"[MaskPropagator] Cutie initialized")
            
        except ImportError as e:
            print(f"[MaskPropagator] Warning: Cutie not available: {e}")
            print("[MaskPropagator] Mask propagation will be disabled")
            self._cutie = False  # Mark as unavailable
    
    def _image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized torch tensor."""
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize
        image = image.astype(np.float32) / 255.0
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        # Add batch dimension
        image = torch.from_numpy(image).unsqueeze(0)
        return image.to(self.device)
    
    def _mask_to_one_hot(self, mask: np.ndarray, num_objects: int) -> torch.Tensor:
        """Convert label mask to one-hot tensor."""
        h, w = mask.shape
        one_hot = torch.zeros((num_objects, h, w), device=self.device)
        
        for i in range(1, num_objects + 1):
            one_hot[i - 1] = torch.from_numpy((mask == i).astype(np.float32))
        
        return one_hot
    
    def _get_overlap_ratio(
        self,
        bbox: np.ndarray,
        other_bboxes: List[np.ndarray]
    ) -> float:
        """Calculate overlap ratio with bboxes that have lower bottom edge."""
        if len(other_bboxes) == 0:
            return 0.0
        
        # bbox in tlwh format
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        area = w * h
        
        if area == 0:
            return 0.0
        
        # Filter boxes with lower bottom edge (likely in front/occluding)
        lower_boxes = []
        for other in other_bboxes:
            ox1, oy1, ow, oh = other
            if (oy1 + oh) > y2:  # Lower bottom edge
                lower_boxes.append(other)
        
        if len(lower_boxes) == 0:
            return 0.0
        
        # Calculate max overlap
        max_overlap = 0.0
        for other in lower_boxes:
            ox1, oy1, ow, oh = other
            ox2, oy2 = ox1 + ow, oy1 + oh
            
            # Intersection
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                overlap = intersection / area
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def initialize_masks(
        self,
        image: np.ndarray,
        bboxes: List[np.ndarray],
        track_ids: List[int],
        overlap_threshold: float = 0.6
    ) -> None:
        """
        Initialize masks for new tracks using SAM.
        
        Args:
            image: Current frame (BGR)
            bboxes: List of bounding boxes in tlwh format
            track_ids: Corresponding track IDs
            overlap_threshold: Skip masks for heavily overlapped objects
        """
        self._init_sam()
        self._init_cutie()
        
        if self._cutie is False:
            return
        
        # Filter heavily overlapped objects
        valid_boxes = []
        valid_ids = []
        
        for bbox, tid in zip(bboxes, track_ids):
            overlap = self._get_overlap_ratio(bbox, bboxes)
            if overlap < overlap_threshold:
                valid_boxes.append(bbox)
                valid_ids.append(tid)
            else:
                self.awaiting_masks.append(tid)
        
        if len(valid_boxes) == 0:
            return
        
        # Convert to xyxy format for SAM
        self._sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        xyxy_boxes = []
        for bbox in valid_boxes:
            x, y, w, h = bbox
            xyxy_boxes.append([x, y, x + w, y + h])
        
        boxes_tensor = torch.tensor(xyxy_boxes, device=self.device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            boxes_tensor, image.shape[:2]
        )
        
        # Generate masks
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Combine masks into single label map
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.int32)
        
        for i, (mask, tid) in enumerate(zip(masks, valid_ids)):
            mask_np = mask.cpu().numpy().squeeze().astype(np.int32)
            mask_id = i + 1
            
            # Avoid overlap with existing masks
            free_pixels = combined_mask == 0
            combined_mask[free_pixels & (mask_np > 0)] = mask_id
            
            # Update state
            self.state.track_to_mask[tid] = mask_id
            self.state.mask_colors[mask_id] = mask_id
        
        # Initialize Cutie with masks
        image_torch = self._image_to_torch(image)
        num_objects = len(valid_ids)
        mask_torch = self._mask_to_one_hot(combined_mask, num_objects)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self._processor.step(image_torch, mask_torch, idx_mask=False)
        
        self.mask_id_counter = num_objects
        self.initialized = True
        print(f"[MaskPropagator] Initialized {num_objects} masks")
    
    def propagate(self, image: np.ndarray) -> MaskState:
        """
        Propagate masks to current frame.
        
        Args:
            image: Current frame (BGR)
            
        Returns:
            MaskState with propagated prediction
        """
        if not self.initialized or self._cutie is False:
            return self.state
        
        self.frame_count += 1
        
        image_torch = self._image_to_torch(image)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                prediction = self._processor.step(image_torch)
        
        # Convert prediction to mask
        if prediction is not None:
            pred_np = prediction.cpu().numpy()
            # Get argmax for each pixel
            mask = np.argmax(pred_np, axis=0)
            self.state.prediction = mask
            
            # Calculate confidence per mask
            for tid, mid in self.state.track_to_mask.items():
                if mid < pred_np.shape[0]:
                    mask_pixels = mask == mid
                    if mask_pixels.sum() > 0:
                        self.state.confidence[tid] = float(pred_np[mid][mask_pixels].mean())
        
        return self.state
    
    def add_masks(
        self,
        image: np.ndarray,
        new_bboxes: List[np.ndarray],
        new_track_ids: List[int],
        all_bboxes: List[np.ndarray],
        overlap_threshold: float = 0.6
    ) -> None:
        """
        Add masks for new tracks.
        
        Args:
            image: Current frame (BGR)
            new_bboxes: Bounding boxes for new tracks
            new_track_ids: Track IDs for new masks
            all_bboxes: All current bounding boxes (for overlap check)
            overlap_threshold: Skip heavily overlapped objects
        """
        if not self.initialized or self._cutie is False:
            return
        
        # Check awaiting masks from previous frames
        # (implement if needed for your use case)
        
        # Filter overlapped
        valid_boxes = []
        valid_ids = []
        
        for bbox, tid in zip(new_bboxes, new_track_ids):
            if tid in self.state.track_to_mask:
                continue
                
            overlap = self._get_overlap_ratio(bbox, all_bboxes)
            if overlap < overlap_threshold:
                valid_boxes.append(bbox)
                valid_ids.append(tid)
            else:
                self.awaiting_masks.append(tid)
        
        if len(valid_boxes) == 0:
            return
        
        # Generate masks with SAM
        self._sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        xyxy_boxes = []
        for bbox in valid_boxes:
            x, y, w, h = bbox
            xyxy_boxes.append([x, y, x + w, y + h])
        
        boxes_tensor = torch.tensor(xyxy_boxes, device=self.device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            boxes_tensor, image.shape[:2]
        )
        
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Add new masks to Cutie
        h, w = image.shape[:2]
        new_mask = np.zeros((h, w), dtype=np.int32)
        new_object_ids = []
        
        for mask, tid in zip(masks, valid_ids):
            mask_np = mask.cpu().numpy().squeeze().astype(np.int32)
            self.mask_id_counter += 1
            mask_id = self.mask_id_counter
            
            new_mask[mask_np > 0] = mask_id
            self.state.track_to_mask[tid] = mask_id
            self.state.mask_colors[mask_id] = mask_id
            new_object_ids.append(mask_id)
        
        if len(new_object_ids) > 0:
            image_torch = self._image_to_torch(image)
            new_mask_torch = torch.from_numpy(new_mask).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    self._processor.step(
                        image_torch, 
                        new_mask_torch, 
                        objects=new_object_ids,
                        idx_mask=True
                    )
    
    def remove_masks(self, track_ids: List[int]) -> None:
        """
        Remove masks for deleted tracks.
        
        Args:
            track_ids: Track IDs to remove
        """
        if not self.initialized or self._cutie is False:
            return
        
        mask_ids_to_remove = []
        for tid in track_ids:
            if tid in self.state.track_to_mask:
                mask_ids_to_remove.append(self.state.track_to_mask[tid])
                del self.state.track_to_mask[tid]
                if tid in self.state.confidence:
                    del self.state.confidence[tid]
        
        if len(mask_ids_to_remove) > 0 and hasattr(self._processor, 'object_manager'):
            self._processor.object_manager.purge_selected_objects(mask_ids_to_remove)
    
    def get_mask_iou(
        self,
        track_id: int,
        bbox: np.ndarray,
    ) -> float:
        """
        Get IoU between track's mask and a bounding box.
        
        Args:
            track_id: Track ID
            bbox: Bounding box in tlwh format
            
        Returns:
            IoU value (0 if no mask)
        """
        if track_id not in self.state.track_to_mask:
            return 0.0
        
        if self.state.prediction is None:
            return 0.0
        
        mask_id = self.state.track_to_mask[track_id]
        mask = self.state.prediction == mask_id
        
        # Create bbox mask
        x, y, w, h = bbox.astype(int)
        h_img, w_img = mask.shape
        
        bbox_mask = np.zeros_like(mask)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        bbox_mask[y1:y2, x1:x2] = True
        
        # Compute IoU
        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()
        
        return intersection / max(union, 1)
    
    def reset(self) -> None:
        """Reset mask propagation state."""
        self.state = MaskState()
        self.frame_count = 0
        self.initialized = False
        self.mask_id_counter = 0
        self.awaiting_masks = []
        
        if self._processor is not None:
            self._processor.clear_memory()


def create_mask_propagator(config) -> MaskPropagator:
    """
    Factory function to create mask propagator from config.
    
    Args:
        config: MaskConfig object
        
    Returns:
        Configured MaskPropagator instance
    """
    return MaskPropagator(
        cutie_weights=config.cutie_weights,
        sam_weights=config.sam_weights,
        sam_model_type=config.sam_model_type,
        device=config.device,
        max_internal_size=config.max_internal_size,
        mem_every=config.mem_every,
    )
