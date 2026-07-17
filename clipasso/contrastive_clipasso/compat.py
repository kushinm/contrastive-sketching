"""
Compatibility shims for dependency version differences.

Handles:
  - Pillow 10+ (BICUBIC moved to Resampling enum)
  - numpy 1.x vs 2.x (deprecated aliases)
  - scikit-image API changes
"""

import PIL
from PIL import Image

# ─── Pillow resampling ────────────────────────────────────────────────
# Pillow < 10:  PIL.Image.BICUBIC, PIL.Image.BILINEAR, etc.
# Pillow >= 10: PIL.Image.Resampling.BICUBIC (old names are deprecated aliases)
#
# We expose a single RESAMPLING dict so the rest of the codebase
# never touches PIL.Image.BICUBIC directly.

if hasattr(Image, "Resampling"):
    # Pillow >= 9.1 (Resampling enum introduced in 9.1, old aliases deprecated in 10)
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
    LANCZOS = Image.Resampling.LANCZOS
else:
    # Pillow < 9.1
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS


# ─── numpy deprecations ──────────────────────────────────────────────
# numpy 1.24+ deprecated np.bool, np.int, np.float, np.complex, np.object, np.str
# numpy 2.0 removed them entirely.
# We don't use these directly but some transitive deps (older scikit-image) might.
# This monkey-patches them back if missing.

import numpy as np

_NUMPY_ALIASES = {
    "bool": np.bool_,
    "int": np.int_,
    "float": np.float64,
    "complex": np.complex128,
    "object": np.object_,
    "str": np.str_,
}

for alias, real_type in _NUMPY_ALIASES.items():
    if not hasattr(np, alias):
        setattr(np, alias, real_type)


# ─── torch version checks ────────────────────────────────────────────

def check_torch_version():
    """Verify PyTorch version and CUDA availability at import time."""
    import torch
    
    major, minor = int(torch.__version__.split('.')[0]), int(torch.__version__.split('.')[1])
    
    if major < 2:
        import warnings
        warnings.warn(
            f"PyTorch {torch.__version__} is older than recommended (2.4+). "
            f"Some features may not work correctly.",
            UserWarning,
        )
    
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ─── diffvg availability ─────────────────────────────────────────────

_DIFFVG_AVAILABLE = None
_DIFFVG_GPU = None

def check_diffvg():
    """Check if pydiffvg is available and whether it supports GPU."""
    global _DIFFVG_AVAILABLE, _DIFFVG_GPU
    
    if _DIFFVG_AVAILABLE is not None:
        return _DIFFVG_AVAILABLE, _DIFFVG_GPU
    
    try:
        import pydiffvg
        _DIFFVG_AVAILABLE = True
        
        # Test GPU support by actually trying to create a Scene with use_gpu=True.
        # pydiffvg.set_use_gpu(True) only sets a flag — the real check happens
        # inside diffvg.Scene() which raises RuntimeError if the C extension
        # was not compiled with CUDA (the common case on CUDA 12).
        import torch
        _DIFFVG_GPU = False
        if torch.cuda.is_available():
            try:
                import diffvg
                filt = diffvg.Filter(diffvg.FilterType.box, 0.5)
                # Minimal scene — triggers the GPU compilation check in C++
                diffvg.Scene(1, 1, [], [], filt, True, 0)
                _DIFFVG_GPU = True
            except (RuntimeError, Exception):
                _DIFFVG_GPU = False
        
        pydiffvg.set_use_gpu(_DIFFVG_GPU)
            
    except ImportError:
        _DIFFVG_AVAILABLE = False
        _DIFFVG_GPU = False
    
    return _DIFFVG_AVAILABLE, _DIFFVG_GPU