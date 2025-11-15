from typing import Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "conversion_results")
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")
    DEBUG_DATA_FOLDER: str = os.path.join(BASE_DIR, "debug_data")
    ARTIFACT_URL: str = "https://models.datalab.to/artifacts"
    FONT_NAME: str = "GoNotoCurrent-Regular.ttf"
    FONT_PATH: str = os.path.join(FONT_DIR, FONT_NAME)
    LOGLEVEL: str = "INFO"

    # General
    OUTPUT_ENCODING: str = "utf-8"
    OUTPUT_IMAGE_FORMAT: str = "JPEG"

    # LLM
    GOOGLE_API_KEY: Optional[str] = ""

    # LLM Service Configuration
    LLM_SERVICE: Optional[str] = None  # Explicit LLM service selection

    # Gemini/Google
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: Optional[str] = None
    THINKING_BUDGET: Optional[int] = None

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_IMAGE_FORMAT: Optional[str] = None

    # Claude/Anthropic
    CLAUDE_API_KEY: Optional[str] = None
    CLAUDE_MODEL_NAME: Optional[str] = None
    MAX_CLAUDE_TOKENS: Optional[int] = None

    # Azure OpenAI
    AZURE_ENDPOINT: Optional[str] = None
    AZURE_API_KEY: Optional[str] = None
    AZURE_API_VERSION: Optional[str] = None
    DEPLOYMENT_NAME: Optional[str] = None

    # Ollama
    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: Optional[str] = None

    # LLM Request Configuration
    TIMEOUT: Optional[int] = None
    MAX_RETRIES: Optional[int] = None
    RETRY_WAIT_TIME: Optional[int] = None
    MAX_OUTPUT_TOKENS: Optional[int] = None

    # General models
    TORCH_DEVICE: Optional[str] = (
        None  # Note: MPS device does not work for text detection, and will default to CPU
    )
    FREE_VRAM_ON_IDLE: bool = False  # Unload models after processing to free VRAM (slower first request)

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cuda":
            return torch.bfloat16
        else:
            return torch.float32

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
