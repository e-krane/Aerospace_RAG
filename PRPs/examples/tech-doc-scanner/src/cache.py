"""
Model caching module for efficient resource management.

Provides singleton caching for expensive models like Pix2Text to avoid
reloading between conversions.
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import Pix2Text with availability check
try:
    from pix2text import Pix2Text

    PIX2TEXT_AVAILABLE = True
except ImportError:
    PIX2TEXT_AVAILABLE = False
    Pix2Text = None


class ModelCache:
    """
    Thread-safe singleton cache for ML models.

    This cache stores loaded models to avoid expensive reloading between
    conversions. Models are cached by device type (cpu, cuda).

    Example:
        >>> cache = ModelCache.get_instance()
        >>> model = cache.get_pix2text_model('cuda')
        >>> # Model is now cached and will be reused
    """

    _instance: Optional["ModelCache"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize empty model cache."""
        if ModelCache._instance is not None:
            raise RuntimeError(
                "ModelCache is a singleton. Use ModelCache.get_instance() instead."
            )

        self._pix2text_models: Dict[str, Pix2Text] = {}
        self._model_lock = threading.Lock()
        logger.info("ModelCache initialized")

    @classmethod
    def get_instance(cls) -> "ModelCache":
        """
        Get singleton instance of ModelCache.

        Returns:
            ModelCache singleton instance

        Example:
            >>> cache = ModelCache.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_pix2text_model(self, device: str = "cpu") -> Pix2Text:
        """
        Get cached Pix2Text model or load new one.

        Args:
            device: Device for model (cpu or cuda)

        Returns:
            Pix2Text model instance

        Raises:
            ImportError: If pix2text is not installed
            ValueError: If device is invalid

        Example:
            >>> cache = ModelCache.get_instance()
            >>> model = cache.get_pix2text_model('cuda')
        """
        if not PIX2TEXT_AVAILABLE:
            raise ImportError(
                "pix2text is required but not installed. Install with: pip install pix2text"
            )

        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

        with self._model_lock:
            if device not in self._pix2text_models:
                logger.info(f"Loading Pix2Text model on device: {device}")
                self._pix2text_models[device] = Pix2Text.from_config(device=device)
                logger.info(f"Pix2Text model loaded and cached for device: {device}")
            else:
                logger.debug(f"Using cached Pix2Text model for device: {device}")

            return self._pix2text_models[device]

    def clear_pix2text_cache(self, device: Optional[str] = None) -> None:
        """
        Clear Pix2Text model cache.

        Args:
            device: Specific device to clear, or None to clear all

        Example:
            >>> cache = ModelCache.get_instance()
            >>> cache.clear_pix2text_cache('cuda')  # Clear cuda model only
            >>> cache.clear_pix2text_cache()  # Clear all models
        """
        with self._model_lock:
            if device is None:
                count = len(self._pix2text_models)
                self._pix2text_models.clear()
                logger.info(f"Cleared {count} Pix2Text model(s) from cache")
            elif device in self._pix2text_models:
                del self._pix2text_models[device]
                logger.info(f"Cleared Pix2Text model for device: {device}")
            else:
                logger.warning(f"No Pix2Text model cached for device: {device}")

    def clear_all(self) -> None:
        """
        Clear all cached models.

        Example:
            >>> cache = ModelCache.get_instance()
            >>> cache.clear_all()
        """
        with self._model_lock:
            total = len(self._pix2text_models)
            self._pix2text_models.clear()
            logger.info(f"Cleared all models from cache (total: {total})")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached models.

        Returns:
            Dictionary with cache statistics

        Example:
            >>> cache = ModelCache.get_instance()
            >>> info = cache.get_cache_info()
            >>> print(f"Cached models: {info['pix2text_models']}")
        """
        with self._model_lock:
            return {"pix2text_models": len(self._pix2text_models)}

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (mainly for testing).

        Warning:
            This will clear all cached models. Use with caution.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear_all()
                cls._instance = None
                logger.info("ModelCache instance reset")
