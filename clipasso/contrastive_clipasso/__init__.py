"""Contrastive CLIPasso: CLIP-guided sketch generation with optional contrastive mode."""

__version__ = "0.1.0"

# Import compat shim early to patch numpy aliases before anything else imports numpy
from . import compat as _compat  # noqa: F401
