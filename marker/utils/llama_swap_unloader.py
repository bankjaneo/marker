import requests
from marker.logger import get_logger
from marker.settings import settings

logger = get_logger()


def unload_llama_swap_model() -> bool:
    """
    Unload the llama-swap model to free up VRAM before OCR processing.

    This function checks if UNLOAD_LLAMA_SWAP_BASE_URL is configured.
    If set, it makes a POST request to the /models/unload endpoint
    to unload the model from the llama-swap service.

    Returns:
        bool: True if unload was successful or not configured, False if unload failed
    """
    # If UNLOAD_LLAMA_SWAP_BASE_URL is not configured, skip unload
    if not settings.UNLOAD_LLAMA_SWAP_BASE_URL:
        return True

    try:
        url = f"{settings.UNLOAD_LLAMA_SWAP_BASE_URL}/models/unload"
        headers = {"Content-Type": "application/json"}

        logger.info(f"Unloading llama-swap model from {url}")
        response = requests.post(url, headers=headers, timeout=10)
        response.raise_for_status()

        logger.info("Successfully unloaded llama-swap model")
        return True

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout while unloading llama-swap model from {settings.UNLOAD_LLAMA_SWAP_BASE_URL}")
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to unload llama-swap model: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error while unloading llama-swap model: {e}")
        return False
