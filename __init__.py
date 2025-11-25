"""
MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space

ICCV 2025

This package provides tools and models for streaming text-to-motion generation.
"""

__version__ = "1.0.0"

# Note: We don't import everything at the package level to avoid import errors
# Users should import from submodules as needed, e.g.:
# from motion_streamer.models import tae
# from motion_streamer.models.llama_model import LLaMAHF

__all__ = [
    "__version__",
]
