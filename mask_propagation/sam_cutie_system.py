"""
Robust SAM + CUTIE Mask Propagation with Identity Preservation
===============================================================
This is the PRODUCTION implementation that properly integrates:
- SAM2 (Segment Anything Model 2) for initial mask generation
- CUTIE for temporal mask propagation
- Mask-based identity verification

Key Features:
1. Generates high-quality masks from detection boxes using SAM
2. Propagates masks temporally using CUTIE's video object segmentation
3. Uses mask overlap for identity verification during occlusion recovery
4. Maintains mask memory for lost tracks

This is designed to preserve player identity even through heavy occlusions.
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class MaskInfo:
    """Information about a tracked mask."""
    track_id: int
    object_id: int              # ID in propagation engine
    mask: Optional[np.ndarray]
    confidence: float = 1.0
    frame_id: int = 0
    bbox: Optional[np.ndarray] = None
    area: float = 0.0
    centroid: Optional[np.ndarray] = None
    
    # History for quality estimation
    area_history: List[float] = field(default_factory=list)
    iou_history: List[float] = field(default_factory=list)
    
    def update_stats(self, new_mask: np.ndarray, frame_id: int):
        """Update mask statistics."""
        self.mask = new_mask
        self.frame_id = frame_id
        self.area = float(new_mask.sum())
        
        # Compute centroid
        if self.area > 0:
            ys, xs = np.where(new_mask > 0)
            self.centroid = np.array([xs.mean(), ys.mean()])
        
        # Update history
        self.area_history.append(self.area)
        if len(self.area_history) > 30:
            self.area_history.pop(0)
        
        # Estimate confidence from area stability
        if len(self.area_history) > 3:
            areas = np.array(self.area_history[-5:])
            cv = np.std(areas) / (np.mean(areas) + 1e-6)
            self.confidence = max(0.1, 1.0 - cv)


class SAMSegmentor:
    """
    SAM-based mask generation from bounding boxes.
    
    Supports both SAM1 and SAM2.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model_type: "vit_b", "vit_l", "vit_h" for SAM1
                       "sam2_tiny", "sam2_small", "sam2_base", "sam2_large" for SAM2
            checkpoint_path: Path to model weights
            device: Device for inference
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        self._model = None
        self._predictor = None
        self._loaded = False
        self._current_image = None
    
    def _load_model(self):
        """Lazy load SAM model."""
        if self._loaded:
            return
        
        # Try SAM2 first
        if 'sam2' in self.model_type.lower():
            self._load_sam2()
        else:
            self._load_sam1()
        
        self._loaded = True
    
    def _load_sam1(self):
        """Load SAM1 model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if self.checkpoint_path is None:
                warnings.warn("SAM checkpoint not provided")
                return
            
            self._model = sam_model_registry[self.model_type](
                checkpoint=self.checkpoint_path
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._predictor = SamPredictor(self._model)
            print(f"[SAM] Loaded SAM1 {self.model_type}")
            
        except ImportError:
            warnings.warn("segment_anything not installed")
        except Exception as e:
            warnings.warn(f"Failed to load SAM1: {e}")
    
    def _load_sam2(self):
        """Load SAM2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            if self.checkpoint_path is None:
                warnings.warn("SAM2 checkpoint not provided")
                return
            
            # Map model type to config
            config_map = {
                "sam2_tiny": "sam2_hiera_t.yaml",
                "sam2_small": "sam2_hiera_s.yaml", 
                "sam2_base": "sam2_hiera_b+.yaml",
                "sam2_large": "sam2_hiera_l.yaml"
            }
            
            config = config_map.get(self.model_type, "sam2_hiera_b+.yaml")
            
            self._model = build_sam2(config, self.checkpoint_path, device=self.device)
            self._predictor = SAM2ImagePredictor(self._model)
            print(f"[SAM] Loaded SAM2 {self.model_type}")
            
        except ImportError:
            warnings.warn("sam2 not installed, falling back to SAM1")
            self._load_sam1()
        except Exception as e:
            warnings.warn(f"Failed to load SAM2: {e}, falling back to SAM1")
            self._load_sam1()
    
    def set_image(self, frame: np.ndarray):
        """Set image for mask prediction."""
        self._load_model()
        
        if self._predictor is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        self._current_image = frame
    
    def segment_box(
        self,
        bbox: np.ndarray,
        multimask: bool = True
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Generate mask from bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            multimask: Return multiple masks and pick best
            
        Returns:
            (mask, confidence) or (None, 0.0)
        """
        if self._predictor is None:
            return None, 0.0
        
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            input_box = np.array([[x1, y1, x2, y2]])
            
            masks, scores, _ = self._predictor.predict(
                box=input_box,
                multimask_output=multimask
            )
            
            if masks is None or len(masks) == 0:
                return None, 0.0
            
            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx].astype(np.uint8)
            confidence = float(scores[best_idx])
            
            return mask, confidence
            
        except Exception as e:
            warnings.warn(f"SAM prediction failed: {e}")
            return None, 0.0
    
    def segment_point(
        self,
        point: np.ndarray,
        label: int = 1
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Generate mask from point prompt.
        
        Args:
            point: [x, y] point coordinates
            label: 1 for foreground, 0 for background
            
        Returns:
            (mask, confidence)
        """
        if self._predictor is None:
            return None, 0.0
        
        try:
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([label])
            
            masks, scores, _ = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8), float(scores[best_idx])
            
        except Exception as e:
            return None, 0.0


class CUTIEPropagator:
    """
    CUTIE-based temporal mask propagation.
    
    Propagates masks across video frames maintaining object identity.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        max_internal_size: int = 480,
        mem_every: int = 5,
        max_mem_frames: int = 10
    ):
        """
        Args:
            checkpoint_path: Path to CUTIE weights
            device: Device for inference
            max_internal_size: Maximum internal resolution
            mem_every: Memory update interval
            max_mem_frames: Maximum memory frames
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_internal_size = max_internal_size
        self.mem_every = mem_every
        self.max_mem_frames = max_mem_frames
        
        self._processor = None
        self._loaded = False
        self._objects = set()
        self._next_object_id = 1
    
    def _load_model(self):
        """Lazy load CUTIE."""
        if self._loaded:
            return
        
        try:
            from cutie.inference.inference_core import InferenceCore
            from cutie.utils.get_default_model import get_default_model
            from omegaconf import DictConfig
            
            # Get config
            if self.checkpoint_path:
                cfg = DictConfig({
                    'weights': self.checkpoint_path,
                    'max_internal_size': self.max_internal_size,
                    'mem_every': self.mem_every,
                    'max_mem_frames': self.max_mem_frames
                })
            else:
                cfg = get_default_model()
                cfg.max_internal_size = self.max_internal_size
                cfg.mem_every = self.mem_every
                cfg.max_mem_frames = self.max_mem_frames
            
            self._processor = InferenceCore(cfg, device=self.device)
            self._loaded = True
            print("[CUTIE] Loaded CUTIE propagator")
            
        except ImportError:
            warnings.warn("cutie not installed")
        except Exception as e:
            warnings.warn(f"Failed to load CUTIE: {e}")
    
    def initialize_object(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        object_id: Optional[int] = None
    ) -> int:
        """
        Initialize new object in CUTIE.
        
        Args:
            frame: Current frame (BGR)
            mask: Binary mask for the object
            object_id: Optional specific ID to use
            
        Returns:
            Assigned object ID
        """
        self._load_model()
        
        if self._processor is None:
            return -1
        
        if object_id is None:
            object_id = self._next_object_id
            self._next_object_id += 1
        
        try:
            # Prepare tensors
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Create labeled mask (each pixel = object_id or 0)
            mask_labeled = mask.astype(np.int64) * object_id
            mask_tensor = torch.from_numpy(mask_labeled).unsqueeze(0).to(self.device)
            
            # Add to CUTIE
            self._processor.step(frame_tensor, mask_tensor, objects=[object_id])
            self._objects.add(object_id)
            
            return object_id
            
        except Exception as e:
            warnings.warn(f"CUTIE initialization failed: {e}")
            return -1
    
    def propagate(
        self,
        frame: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Propagate masks to new frame.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            Dictionary mapping object_id to mask
        """
        if self._processor is None or len(self._objects) == 0:
            return {}
        
        try:
            # Prepare frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Propagate
            with torch.no_grad():
                output = self._processor.step(frame_tensor)
            
            if output is None:
                return {}
            
            # Extract per-object masks
            pred_mask = output.argmax(dim=1)[0].cpu().numpy()
            
            masks = {}
            for obj_id in self._objects:
                obj_mask = (pred_mask == obj_id).astype(np.uint8)
                if obj_mask.sum() > 0:
                    masks[obj_id] = obj_mask
            
            return masks
            
        except Exception as e:
            warnings.warn(f"CUTIE propagation failed: {e}")
            return {}
    
    def remove_object(self, object_id: int):
        """Remove object from tracking."""
        self._objects.discard(object_id)
    
    def reset(self):
        """Reset propagator state."""
        if self._processor is not None:
            try:
                self._processor.clear_memory()
            except:
                pass
        self._objects.clear()
        self._next_object_id = 1


class MaskIdentitySystem:
    """
    Complete mask-based identity preservation system.
    
    Integrates SAM and CUTIE for:
    1. Initial mask generation from detection boxes
    2. Temporal mask propagation
    3. Identity verification using mask overlap
    4. Lost track recovery using mask memory
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        cutie_checkpoint: Optional[str] = None,
        device: str = "cuda",
        min_mask_area: float = 100,
        min_iou_threshold: float = 0.3
    ):
        """
        Args:
            sam_checkpoint: Path to SAM weights
            sam_model_type: SAM model variant
            cutie_checkpoint: Path to CUTIE weights
            device: Device for inference
            min_mask_area: Minimum valid mask area
            min_iou_threshold: Minimum IoU for identity match
        """
        self.device = device
        self.min_mask_area = min_mask_area
        self.min_iou_threshold = min_iou_threshold
        
        # Initialize components
        self.sam = SAMSegmentor(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            device=device
        )
        
        self.cutie = CUTIEPropagator(
            checkpoint_path=cutie_checkpoint,
            device=device
        )
        
        # Track-to-mask mapping
        self.track_masks: Dict[int, MaskInfo] = {}
        self._track_to_obj: Dict[int, int] = {}
        self._obj_to_track: Dict[int, int] = {}
        
        # Lost track memory
        self.lost_masks: Dict[int, MaskInfo] = {}
        
        # Frame counter
        self.frame_id = 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        track_bboxes: Dict[int, np.ndarray],
        new_track_ids: List[int] = None,
        lost_track_ids: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Process a frame: propagate existing masks and init new ones.
        
        Args:
            frame: Current frame (BGR)
            track_bboxes: {track_id: [x1,y1,x2,y2]} for all active tracks
            new_track_ids: IDs of newly created tracks
            lost_track_ids: IDs of tracks that were lost this frame
            
        Returns:
            {track_id: mask} for all tracks
        """
        self.frame_id += 1
        new_track_ids = new_track_ids or []
        lost_track_ids = lost_track_ids or []
        
        # Store lost track masks for recovery
        for tid in lost_track_ids:
            if tid in self.track_masks:
                self.lost_masks[tid] = self.track_masks[tid]
        
        # Set SAM image
        self.sam.set_image(frame)
        
        # Initialize masks for new tracks
        for tid in new_track_ids:
            if tid in track_bboxes:
                self._init_track_mask(frame, tid, track_bboxes[tid])
        
        # Propagate existing masks
        propagated = self.cutie.propagate(frame)
        
        # Update track masks with propagated results
        result_masks = {}
        
        for tid, mask_info in list(self.track_masks.items()):
            obj_id = self._track_to_obj.get(tid)
            
            if obj_id in propagated:
                # Got propagated mask
                prop_mask = propagated[obj_id]
                
                # Validate propagation quality
                if self._validate_propagated_mask(mask_info, prop_mask, track_bboxes.get(tid)):
                    mask_info.update_stats(prop_mask, self.frame_id)
                    result_masks[tid] = prop_mask
                else:
                    # Propagation failed, re-init from detection
                    if tid in track_bboxes:
                        self._reinit_track_mask(frame, tid, track_bboxes[tid])
                        if tid in self.track_masks:
                            result_masks[tid] = self.track_masks[tid].mask
            else:
                # No propagation, try to re-init
                if tid in track_bboxes:
                    self._reinit_track_mask(frame, tid, track_bboxes[tid])
                    if tid in self.track_masks and self.track_masks[tid].mask is not None:
                        result_masks[tid] = self.track_masks[tid].mask
        
        return result_masks
    
    def _init_track_mask(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray
    ):
        """Initialize mask for new track using SAM."""
        mask, confidence = self.sam.segment_box(bbox)
        
        if mask is None or mask.sum() < self.min_mask_area:
            return
        
        # Initialize in CUTIE
        obj_id = self.cutie.initialize_object(frame, mask)
        
        if obj_id < 0:
            return
        
        # Store mapping
        self._track_to_obj[track_id] = obj_id
        self._obj_to_track[obj_id] = track_id
        
        # Create mask info
        mask_info = MaskInfo(
            track_id=track_id,
            object_id=obj_id,
            mask=mask,
            confidence=confidence,
            frame_id=self.frame_id,
            bbox=bbox.copy()
        )
        mask_info.update_stats(mask, self.frame_id)
        
        self.track_masks[track_id] = mask_info
    
    def _reinit_track_mask(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray
    ):
        """Re-initialize mask for existing track."""
        mask, confidence = self.sam.segment_box(bbox)
        
        if mask is None or mask.sum() < self.min_mask_area:
            return
        
        # Get existing object ID or create new
        obj_id = self._track_to_obj.get(track_id)
        
        if obj_id is None:
            obj_id = self.cutie.initialize_object(frame, mask)
            if obj_id < 0:
                return
            self._track_to_obj[track_id] = obj_id
            self._obj_to_track[obj_id] = track_id
        else:
            # Re-initialize existing object
            self.cutie.initialize_object(frame, mask, obj_id)
        
        # Update mask info
        if track_id in self.track_masks:
            self.track_masks[track_id].mask = mask
            self.track_masks[track_id].confidence = confidence
            self.track_masks[track_id].update_stats(mask, self.frame_id)
        else:
            mask_info = MaskInfo(
                track_id=track_id,
                object_id=obj_id,
                mask=mask,
                confidence=confidence,
                frame_id=self.frame_id,
                bbox=bbox.copy()
            )
            mask_info.update_stats(mask, self.frame_id)
            self.track_masks[track_id] = mask_info
    
    def _validate_propagated_mask(
        self,
        prev_info: MaskInfo,
        new_mask: np.ndarray,
        bbox: Optional[np.ndarray]
    ) -> bool:
        """Validate quality of propagated mask."""
        if new_mask is None:
            return False
        
        new_area = new_mask.sum()
        
        # Check minimum area
        if new_area < self.min_mask_area:
            return False
        
        # Check area consistency
        if prev_info.area > 0:
            area_ratio = new_area / prev_info.area
            if area_ratio < 0.3 or area_ratio > 3.0:
                return False
        
        # Check bbox overlap if available
        if bbox is not None:
            # Compute mask bbox
            ys, xs = np.where(new_mask > 0)
            if len(xs) == 0:
                return False
            
            mask_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            
            # Check IoU with detection bbox
            iou = self._bbox_iou(mask_bbox, bbox)
            if iou < 0.2:
                return False
        
        return True
    
    def _bbox_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def compute_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """Compute IoU between two masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        return intersection / (union + 1e-6)
    
    def verify_identity(
        self,
        lost_track_id: int,
        candidate_mask: np.ndarray,
        candidate_bbox: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Verify if candidate matches a lost track.
        
        Args:
            lost_track_id: ID of lost track to check
            candidate_mask: Mask from potential re-detection
            candidate_bbox: Bbox from potential re-detection
            
        Returns:
            (is_match, confidence)
        """
        if lost_track_id not in self.lost_masks:
            return False, 0.0
        
        lost_info = self.lost_masks[lost_track_id]
        
        if lost_info.mask is None:
            return False, 0.0
        
        # Compute mask IoU
        mask_iou = self.compute_mask_iou(lost_info.mask, candidate_mask)
        
        # Position check
        position_valid = True
        if lost_info.bbox is not None:
            bbox_iou = self._bbox_iou(lost_info.bbox, candidate_bbox)
            position_valid = bbox_iou > 0.1
        
        # Identity decision
        is_match = mask_iou >= self.min_iou_threshold and position_valid
        confidence = mask_iou
        
        return is_match, confidence
    
    def compute_mask_cost_matrix(
        self,
        track_ids: List[int],
        detection_bboxes: List[np.ndarray],
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Compute mask-based cost matrix for association.
        
        Args:
            track_ids: List of track IDs
            detection_bboxes: List of detection boxes
            frame: Current frame (for generating detection masks)
            
        Returns:
            Cost matrix (1 - mask_iou)
        """
        n_tracks = len(track_ids)
        n_dets = len(detection_bboxes)
        
        if n_tracks == 0 or n_dets == 0:
            return np.ones((n_tracks, n_dets))
        
        # Generate masks for detections
        self.sam.set_image(frame)
        det_masks = []
        for bbox in detection_bboxes:
            mask, _ = self.sam.segment_box(bbox)
            det_masks.append(mask)
        
        # Compute cost matrix
        cost = np.ones((n_tracks, n_dets))
        
        for i, tid in enumerate(track_ids):
            if tid not in self.track_masks:
                continue
            track_mask = self.track_masks[tid].mask
            
            for j, det_mask in enumerate(det_masks):
                if det_mask is not None and track_mask is not None:
                    iou = self.compute_mask_iou(track_mask, det_mask)
                    cost[i, j] = 1.0 - iou
        
        return cost
    
    def remove_track(self, track_id: int):
        """Remove track from system."""
        if track_id in self.track_masks:
            obj_id = self._track_to_obj.get(track_id)
            if obj_id is not None:
                self.cutie.remove_object(obj_id)
                del self._track_to_obj[track_id]
                self._obj_to_track.pop(obj_id, None)
            del self.track_masks[track_id]
        
        self.lost_masks.pop(track_id, None)
    
    def cleanup(self, active_track_ids: set, max_lost_age: int = 90):
        """Clean up inactive tracks and old lost masks."""
        # Remove inactive tracks
        for tid in list(self.track_masks.keys()):
            if tid not in active_track_ids:
                self.remove_track(tid)
        
        # Remove old lost masks
        for tid in list(self.lost_masks.keys()):
            if self.frame_id - self.lost_masks[tid].frame_id > max_lost_age:
                del self.lost_masks[tid]
    
    def reset(self):
        """Reset entire system."""
        self.cutie.reset()
        self.track_masks.clear()
        self._track_to_obj.clear()
        self._obj_to_track.clear()
        self.lost_masks.clear()
        self.frame_id = 0
    
    def get_mask(self, track_id: int) -> Optional[np.ndarray]:
        """Get current mask for track."""
        if track_id in self.track_masks:
            return self.track_masks[track_id].mask
        return None
    
    def get_all_masks_combined(self) -> np.ndarray:
        """Get combined mask with track IDs as pixel values."""
        if not self.track_masks:
            return None
        
        # Get frame size from first mask
        first_mask = next(iter(self.track_masks.values())).mask
        if first_mask is None:
            return None
        
        combined = np.zeros(first_mask.shape, dtype=np.int32)
        
        for tid, info in self.track_masks.items():
            if info.mask is not None:
                combined[info.mask > 0] = tid
        
        return combined
