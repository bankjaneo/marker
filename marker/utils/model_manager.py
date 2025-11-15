"""
Model manager for dynamic VRAM management.

Handles lazy loading and unloading of ML models based on the FREE_VRAM_ON_IDLE setting.
When enabled, models are loaded on-demand and unloaded after processing to free VRAM.
"""

import gc
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
        """Unload all models from memory and clear GPU cache comprehensively."""
        with self.lock:
            # Step 1: Move models to CPU if using CUDA (helps ensure GPU tensors are released)
            if settings.TORCH_DEVICE_MODEL == "cuda":
                for model_name, model_obj in list(self.models.items()):
                    try:
                        # Try to move predictor models to CPU
                        if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'to'):
                            model_obj.model.to('cpu')
                        elif hasattr(model_obj, 'to'):
                            model_obj.to('cpu')
                    except Exception:
                        # Some models may not support .to(), skip gracefully
                        pass

            # Step 2: Explicitly delete each model object
            for model_name in list(self.models.keys()):
                try:
                    del self.models[model_name]
                except Exception:
                    pass

            # Step 3: Clear the dictionary
            self.models.clear()

            # Step 4: Force Python garbage collection (multiple passes for circular references)
            for _ in range(3):
                gc.collect()

            # Step 5: Clear PyTorch's CUDA cache and synchronize
            if settings.TORCH_DEVICE_MODEL == "cuda":
                torch.cuda.empty_cache()
                # Synchronize to ensure all CUDA operations complete
                torch.cuda.synchronize()

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
