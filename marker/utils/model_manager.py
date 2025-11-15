"""
Model manager for dynamic VRAM management.

Handles lazy loading and unloading of ML models based on the FREE_VRAM_ON_IDLE setting.
When enabled, models are loaded on-demand and unloaded after processing to free VRAM.
"""

import threading
from typing import Dict, Any, Optional
import torch
from marker.models import create_model_dict
from marker.settings import settings


class ModelManager:
    """
    Manages dynamic loading and unloading of ML models for VRAM optimization.

    When FREE_VRAM_ON_IDLE is True:
    - Models are loaded on first access
    - Models are unloaded after processing
    - Subsequent requests trigger reloading (slower but saves VRAM)

    When FREE_VRAM_ON_IDLE is False:
    - Models are pre-loaded at initialization
    - Models remain in VRAM throughout server lifetime (faster, uses more VRAM)
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.lock = threading.RLock()  # Thread-safe for concurrent requests
        self.free_vram_on_idle = settings.FREE_VRAM_ON_IDLE

        # If not using dynamic VRAM management, pre-load all models
        if not self.free_vram_on_idle:
            self._load_all_models()

    def _load_all_models(self):
        """Load all models into memory."""
        with self.lock:
            if not self.models:  # Only load if not already loaded
                self.models = create_model_dict(
                    device=settings.TORCH_DEVICE_MODEL,
                    dtype=settings.MODEL_DTYPE,
                )

    def _unload_all_models(self):
        """Unload all models from memory and clear GPU cache."""
        with self.lock:
            self.models.clear()

            # Force GPU memory cleanup if using CUDA
            if settings.TORCH_DEVICE_MODEL == "cuda":
                torch.cuda.empty_cache()

    def get_models(self) -> Dict[str, Any]:
        """
        Get the models dictionary, loading them if necessary.

        If FREE_VRAM_ON_IDLE is enabled, models are loaded on first access.
        Otherwise, they should already be pre-loaded.

        Returns:
            Dictionary containing all loaded models
        """
        with self.lock:
            if not self.models:
                self._load_all_models()
            return self.models

    def release_models(self):
        """
        Release (unload) models if FREE_VRAM_ON_IDLE is enabled.

        This should be called after processing is complete to free VRAM.
        If FREE_VRAM_ON_IDLE is False, this is a no-op.
        """
        if self.free_vram_on_idle:
            self._unload_all_models()

    def cleanup(self):
        """
        Full cleanup - unload all models regardless of settings.

        Should be called during server shutdown.
        """
        self._unload_all_models()
