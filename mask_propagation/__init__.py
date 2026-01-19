"""
Mask Propagation Module
=======================
SAM + CUTIE integration for identity preservation.
"""

from .integrated_masks import (
    MaskPropagationSystem,
    MaskIdentityVerifier,
    MaskState,
    PropagationResult
)

# Legacy
from .mask_propagator import MaskPropagator

__all__ = [
    'MaskPropagationSystem',
    'MaskIdentityVerifier',
    'MaskState',
    'PropagationResult',
    'MaskPropagator',
]
