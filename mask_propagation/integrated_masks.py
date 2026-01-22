"""
Integrated SAM + CUTIE Mask Propagation System
==============================================
Properly integrates:
- SAM (Segment Anything Model) for initial mask generation
- CUTIE for temporal mask propagation
- Mask-based identity verification
- Occlusion-aware mask management

This is designed for identity preservation through occlusions.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class MaskState:
    """State of mask propagation for a single track."""
    track_id: int
    mask_id: int               # ID in CUTIE
    current_mask: Optional[np.ndarray] = None
    confidence: float = 1.0
    last_update_frame: int = 0
    is_valid: bool = True
    occlusion_frames: int = 0
    area_history: List[float] = field(default_factory=list)


@dataclass  
class PropagationResult:
    """Result of mask propagation for a frame."""
    masks: Dict[int, np.ndarray]  # track_id -> mask
    confidences: Dict[int, float]  # track_id -> confidence
    combined_mask: Optional[np.ndarray] = None  # All masks combined with IDs


class MaskPropagationSystem:
    """
    Integrated mask propagation system for identity-preserving tracking.
    
    Workflow:
    1. New track → SAM generates initial mask from bbox
    2. Frame-by-frame → CUTIE propagates masks temporally
    3. Track lost → Mask kept in memory for re-identification
    4. Track recovered → Mask IoU helps verify identity
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        cutie_checkpoint: Optional[str] = None,
        device: str = "cuda",
        max_objects: int = 50,
        mem_every: int = 5,
        max_mem_frames: int = 10,
    ):
        """
        Initialize mask propagation system.
        
        Args:
            sam_checkpoint: Path to SAM weights
            sam_model_type: SAM model type (vit_b, vit_l, vit_h)
            cutie_checkpoint: Path to CUTIE weights
            device: Device for inference
            max_objects: Maximum tracked objects
            mem_every: CUTIE memory update interval
            max_mem_frames: Maximum memory frames in CUTIE
        """
        self.device = device
        self.max_objects = max_objects
        self.mem_every = mem_every
        self.max_mem_frames = max_mem_frames
        
        # Model paths
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.cutie_checkpoint = cutie_checkpoint
        
        # Lazy-loaded models
        self._sam = None
        self._sam_predictor = None
        self._cutie = None
        self._cutie_processor = None
        
        # Mask state management
        self.mask_states: Dict[int, MaskState] = {}
        self._next_mask_id = 1
        self._track_to_mask: Dict[int, int] = {}
        self._mask_to_track: Dict[int, int] = {}
        
        # Frame tracking
        self.frame_count = 0
        self.last_frame = None
        
        # CUTIE memory management
        self._cutie_initialized = False
        self._cutie_objects = set()
    
    def _load_sam(self):
        """Lazy load SAM model."""
        if self._sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if self.sam_checkpoint is None:
                warnings.warn("SAM checkpoint not provided, mask generation disabled")
                return
            
            self._sam = sam_model_registry[self.sam_model_type](
                checkpoint=self.sam_checkpoint
            )
            
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._sam = self._sam.cuda()
            
            self._sam_predictor = SamPredictor(self._sam)
            print(f"[MaskPropagation] SAM loaded: {self.sam_model_type}")
            
        except ImportError:
            warnings.warn("segment_anything not installed, mask generation disabled")
        except Exception as e:
            warnings.warn(f"Failed to load SAM: {e}")
    
    def _load_cutie(self):
        """Lazy load CUTIE model."""
        if self._cutie is not None:
            return
        
        try:
            import torch
            from cutie.inference.inference_core import InferenceCore
            from cutie.utils.get_default_model import get_default_model
            
            if self.cutie_checkpoint:
                cfg = {
                    'model': {'path': self.cutie_checkpoint},
                    'max_mem_frames': self.max_mem_frames,
                    'mem_every': self.mem_every,
                }
            else:
                # Use default model
                cfg = get_default_model()
                cfg['max_mem_frames'] = self.max_mem_frames
                cfg['mem_every'] = self.mem_every
            
            self._cutie = InferenceCore(cfg, device=self.device)
            print("[MaskPropagation] CUTIE loaded")
            
        except ImportError:
            warnings.warn("cutie not installed, mask propagation disabled")
        except Exception as e:
            warnings.warn(f"Failed to load CUTIE: {e}")
    
    def initialize_track_mask(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray,
        class_id: int = 0
    ) -> Optional[np.ndarray]:
        """
        Initialize mask for a new track using SAM.
        
        Args:
            frame: Current frame (BGR)
            track_id: Track ID
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Object class
            
        Returns:
            Generated mask or None
        """
        self._load_sam()
        
        if self._sam_predictor is None:
            return None
        
        # Set image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(rgb_frame)
        
        # Generate mask from bbox
        x1, y1, x2, y2 = bbox.astype(int)
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = self._sam_predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)
        
        # Assign mask ID
        mask_id = self._next_mask_id
        self._next_mask_id += 1
        
        # Store state
        self.mask_states[track_id] = MaskState(
            track_id=track_id,
            mask_id=mask_id,
            current_mask=mask,
            confidence=float(scores[best_idx]),
            last_update_frame=self.frame_count,
            area_history=[mask.sum()]
        )
        
        self._track_to_mask[track_id] = mask_id
        self._mask_to_track[mask_id] = track_id
        
        # Initialize in CUTIE if available
        self._add_to_cutie(frame, mask, mask_id)
        
        return mask
    
    def _add_to_cutie(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        mask_id: int
    ):
        """Add new object to CUTIE tracker."""
        self._load_cutie()
        
        if self._cutie is None:
            return
        
        try:
            import torch
            
            # Prepare frame tensor
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Prepare mask tensor (with object ID)
            mask_tensor = torch.from_numpy(mask.astype(np.int64)) * mask_id
            mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
            
            if not self._cutie_initialized:
                # First object initializes CUTIE
                self._cutie.set_all_labels(list(range(1, self.max_objects + 1)))
                self._cutie_initialized = True
            
            # Add object to memory
            self._cutie.step(frame_tensor, mask_tensor, objects=[mask_id])
            self._cutie_objects.add(mask_id)
            
        except Exception as e:
            warnings.warn(f"Failed to add object to CUTIE: {e}")
    
    def propagate(
        self,
        frame: np.ndarray,
        active_track_ids: Optional[set] = None
    ) -> PropagationResult:
        """
        Propagate masks to new frame using CUTIE.
        
        Args:
            frame: Current frame (BGR)
            active_track_ids: Set of currently active track IDs
            
        Returns:
            PropagationResult with masks for all tracks
        """
        self.frame_count += 1
        self.last_frame = frame
        
        result = PropagationResult(masks={}, confidences={})
        
        if not self._cutie_initialized or self._cutie is None:
            return result
        
        try:
            import torch
            
            # Prepare frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Propagate
            with torch.no_grad():
                output = self._cutie.step(frame_tensor)
            
            # Extract masks per object
            if output is not None:
                pred_mask = output.argmax(dim=1)[0].cpu().numpy()
                
                # Build combined mask
                result.combined_mask = pred_mask
                
                # Extract individual masks
                for mask_id in self._cutie_objects:
                    if mask_id in self._mask_to_track:
                        track_id = self._mask_to_track[mask_id]
                        
                        if active_track_ids and track_id not in active_track_ids:
                            continue
                        
                        track_mask = (pred_mask == mask_id).astype(np.uint8)
                        
                        if track_mask.sum() > 0:
                            result.masks[track_id] = track_mask
                            
                            # Update state
                            if track_id in self.mask_states:
                                state = self.mask_states[track_id]
                                state.current_mask = track_mask
                                state.last_update_frame = self.frame_count
                                state.area_history.append(track_mask.sum())
                                
                                # Compute confidence based on area consistency
                                if len(state.area_history) > 3:
                                    recent_areas = state.area_history[-5:]
                                    area_std = np.std(recent_areas) / (np.mean(recent_areas) + 1e-6)
                                    state.confidence = max(0.1, 1.0 - area_std)
                                
                                result.confidences[track_id] = state.confidence
            
        except Exception as e:
            warnings.warn(f"CUTIE propagation failed: {e}")
        
        return result
    
    def update_mask(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray,
        regenerate: bool = False
    ) -> Optional[np.ndarray]:
        """
        Update mask for existing track.
        
        Args:
            frame: Current frame
            track_id: Track ID
            bbox: Current bounding box
            regenerate: Force regenerate using SAM
            
        Returns:
            Updated mask
        """
        if track_id not in self.mask_states:
            # New track, initialize
            return self.initialize_track_mask(frame, track_id, bbox)
        
        state = self.mask_states[track_id]
        
        # Check if mask needs regeneration
        should_regenerate = regenerate
        
        if state.current_mask is not None:
            # Check mask quality
            mask_area = state.current_mask.sum()
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            coverage = mask_area / (bbox_area + 1e-6)
            
            # Regenerate if mask is too small or too large
            if coverage < 0.1 or coverage > 2.0:
                should_regenerate = True
            
            # Check for rapid area changes
            if len(state.area_history) > 3:
                recent_change = abs(mask_area - state.area_history[-1]) / (state.area_history[-1] + 1e-6)
                if recent_change > 0.5:  # >50% change
                    state.occlusion_frames += 1
                    if state.occlusion_frames > 5:
                        should_regenerate = True
                else:
                    state.occlusion_frames = 0
        
        if should_regenerate:
            # Regenerate mask using SAM
            new_mask = self._regenerate_mask(frame, track_id, bbox)
            if new_mask is not None:
                state.current_mask = new_mask
                state.confidence = 1.0
                state.occlusion_frames = 0
                
                # Update CUTIE
                self._add_to_cutie(frame, new_mask, state.mask_id)
                
            return new_mask
        
        return state.current_mask
    
    def _regenerate_mask(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Regenerate mask using SAM."""
        self._load_sam()
        
        if self._sam_predictor is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(rgb_frame)
        
        x1, y1, x2, y2 = bbox.astype(int)
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = self._sam_predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)
    
    def remove_track(self, track_id: int):
        """Remove track from mask propagation."""
        if track_id in self.mask_states:
            state = self.mask_states[track_id]
            mask_id = state.mask_id
            
            # Remove from mappings
            if track_id in self._track_to_mask:
                del self._track_to_mask[track_id]
            if mask_id in self._mask_to_track:
                del self._mask_to_track[mask_id]
            
            # Remove from CUTIE
            if mask_id in self._cutie_objects:
                self._cutie_objects.discard(mask_id)
            
            del self.mask_states[track_id]
    
    def get_mask(self, track_id: int) -> Optional[np.ndarray]:
        """Get current mask for track."""
        if track_id in self.mask_states:
            return self.mask_states[track_id].current_mask
        return None
    
    def get_mask_iou(
        self,
        track_id: int,
        other_mask: np.ndarray
    ) -> float:
        """
        Compute IoU between track's mask and another mask.
        Useful for identity verification.
        """
        track_mask = self.get_mask(track_id)
        if track_mask is None:
            return 0.0
        
        if track_mask.shape != other_mask.shape:
            return 0.0
        
        intersection = np.logical_and(track_mask > 0, other_mask > 0).sum()
        union = np.logical_or(track_mask > 0, other_mask > 0).sum()
        
        return intersection / (union + 1e-6)
    
    def compute_mask_cost_matrix(
        self,
        track_ids: List[int],
        detection_masks: List[Optional[np.ndarray]]
    ) -> np.ndarray:
        """
        Compute mask-based cost matrix for association.
        
        Args:
            track_ids: List of track IDs
            detection_masks: List of detection masks (can contain None)
            
        Returns:
            Cost matrix (1 - mask_iou)
        """
        n_tracks = len(track_ids)
        n_dets = len(detection_masks)
        
        cost_matrix = np.ones((n_tracks, n_dets))
        
        for i, tid in enumerate(track_ids):
            track_mask = self.get_mask(tid)
            if track_mask is None:
                continue
            
            for j, det_mask in enumerate(detection_masks):
                if det_mask is None:
                    continue
                
                if track_mask.shape != det_mask.shape:
                    continue
                
                iou = self.get_mask_iou(tid, det_mask)
                cost_matrix[i, j] = 1.0 - iou
        
        return cost_matrix
    
    def reset(self):
        """Reset all mask propagation state."""
        self.mask_states.clear()
        self._track_to_mask.clear()
        self._mask_to_track.clear()
        self._cutie_objects.clear()
        self._next_mask_id = 1
        self.frame_count = 0
        self._cutie_initialized = False
        
        # Reset CUTIE memory
        if self._cutie is not None:
            try:
                self._cutie.clear_memory()
            except:
                pass


class MaskIdentityVerifier:
    """
    Uses masks to verify and preserve track identity.
    
    When a lost track is potentially re-identified:
    1. Compare predicted mask position with detection
    2. Use mask IoU as identity confidence
    3. Only accept re-identification if mask matches
    """
    
    def __init__(
        self,
        mask_system: MaskPropagationSystem,
        min_mask_iou: float = 0.3,
        position_tolerance: float = 100.0
    ):
        """
        Args:
            mask_system: MaskPropagationSystem instance
            min_mask_iou: Minimum mask IoU for identity match
            position_tolerance: Maximum position deviation (pixels)
        """
        self.mask_system = mask_system
        self.min_mask_iou = min_mask_iou
        self.position_tolerance = position_tolerance
        
        # Lost track memory
        self.lost_track_masks: Dict[int, np.ndarray] = {}
        self.lost_track_positions: Dict[int, np.ndarray] = {}
        self.lost_track_velocities: Dict[int, np.ndarray] = {}
        self.lost_frame: Dict[int, int] = {}
    
    def store_lost_track(
        self,
        track_id: int,
        mask: Optional[np.ndarray],
        position: np.ndarray,
        velocity: np.ndarray,
        frame_id: int
    ):
        """Store information about lost track for later re-identification."""
        if mask is not None:
            self.lost_track_masks[track_id] = mask.copy()
        self.lost_track_positions[track_id] = position.copy()
        self.lost_track_velocities[track_id] = velocity.copy()
        self.lost_frame[track_id] = frame_id
    
    def verify_identity(
        self,
        lost_track_id: int,
        detection_mask: Optional[np.ndarray],
        detection_bbox: np.ndarray,
        current_frame: int
    ) -> Tuple[bool, float]:
        """
        Verify if detection matches lost track identity.
        
        Args:
            lost_track_id: ID of lost track
            detection_mask: Mask of detection (from SAM)
            detection_bbox: Bbox of detection [x1, y1, x2, y2]
            current_frame: Current frame number
            
        Returns:
            (is_match, confidence)
        """
        if lost_track_id not in self.lost_track_positions:
            return False, 0.0
        
        # Predict where track should be
        frames_elapsed = current_frame - self.lost_frame[lost_track_id]
        predicted_pos = (
            self.lost_track_positions[lost_track_id] +
            self.lost_track_velocities[lost_track_id] * frames_elapsed
        )
        
        # Detection center
        det_center = np.array([
            (detection_bbox[0] + detection_bbox[2]) / 2,
            (detection_bbox[1] + detection_bbox[3]) / 2
        ])
        
        # Position check
        position_dist = np.linalg.norm(predicted_pos - det_center)
        position_ok = position_dist < self.position_tolerance * (1 + 0.1 * frames_elapsed)
        
        if not position_ok:
            return False, 0.0
        
        # Mask check
        mask_iou = 0.0
        if (lost_track_id in self.lost_track_masks and 
            detection_mask is not None):
            
            lost_mask = self.lost_track_masks[lost_track_id]
            
            if lost_mask.shape == detection_mask.shape:
                intersection = np.logical_and(lost_mask > 0, detection_mask > 0).sum()
                union = np.logical_or(lost_mask > 0, detection_mask > 0).sum()
                mask_iou = intersection / (union + 1e-6)
        
        # Compute confidence
        position_conf = max(0, 1.0 - position_dist / self.position_tolerance)
        mask_conf = mask_iou
        
        # Combine confidences
        if mask_iou > 0:
            confidence = 0.4 * position_conf + 0.6 * mask_conf
            is_match = mask_iou >= self.min_mask_iou
        else:
            confidence = position_conf * 0.7  # Lower confidence without mask
            is_match = position_conf > 0.5
        
        return is_match, confidence
    
    def cleanup(self, active_track_ids: set, max_age: int = 90):
        """Remove old lost tracks."""
        to_remove = []
        
        for tid, frame in self.lost_frame.items():
            if tid in active_track_ids:
                to_remove.append(tid)
            # Remove if too old (use frame count instead of exact age)
            # This is simplified - in real impl, pass current_frame
        
        for tid in to_remove:
            self.lost_track_masks.pop(tid, None)
            self.lost_track_positions.pop(tid, None)
            self.lost_track_velocities.pop(tid, None)
            self.lost_frame.pop(tid, None)
