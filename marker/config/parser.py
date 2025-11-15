import json
import os
from typing import Dict

import click

from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.renderers.chunk import ChunkRenderer
from marker.renderers.html import HTMLRenderer
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.settings import settings
from marker.util import classes_to_strings, parse_range_str, strings_to_classes

logger = get_logger()


class ConfigParser:
    def __init__(self, cli_options: dict):
        self.cli_options = cli_options

    @staticmethod
    def common_options(fn):
        fn = click.option(
            "--output_dir",
            type=click.Path(exists=False),
            required=False,
            default=settings.OUTPUT_DIR,
            help="Directory to save output.",
        )(fn)
        fn = click.option("--debug", "-d", is_flag=True, help="Enable debug mode.")(fn)
        fn = click.option(
            "--output_format",
            type=click.Choice(["markdown", "json", "html", "chunks"]),
            default="markdown",
            help="Format to output results in.",
        )(fn)
        fn = click.option(
            "--processors",
            type=str,
            default=None,
            help="Comma separated list of processors to use.  Must use full module path.",
        )(fn)
        fn = click.option(
            "--config_json",
            type=str,
            default=None,
            help="Path to JSON file with additional configuration.",
        )(fn)
        fn = click.option(
            "--disable_multiprocessing",
            is_flag=True,
            default=False,
            help="Disable multiprocessing.",
        )(fn)
        fn = click.option(
            "--disable_image_extraction",
            is_flag=True,
            default=False,
            help="Disable image extraction.",
        )(fn)
        # these are options that need a list transformation, i.e splitting/parsing a string
        fn = click.option(
            "--page_range",
            type=str,
            default=None,
            help="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
        )(fn)

        # we put common options here
        fn = click.option(
            "--converter_cls",
            type=str,
            default=None,
            help="Converter class to use.  Defaults to PDF converter.",
        )(fn)
        fn = click.option(
            "--llm_service",
            type=str,
            default=None,
            help="LLM service to use - should be full import path, like marker.services.gemini.GoogleGeminiService",
        )(fn)
        return fn

    def generate_config_dict(self) -> Dict[str, any]:
        config = {}
        output_dir = self.cli_options.get("output_dir", settings.OUTPUT_DIR)
        for k, v in self.cli_options.items():
            if not v:
                continue

            match k:
                case "debug":
                    config["debug_pdf_images"] = True
                    config["debug_layout_images"] = True
                    config["debug_json"] = True
                    config["debug_data_folder"] = output_dir
                case "page_range":
                    config["page_range"] = parse_range_str(v)
                case "config_json":
                    with open(v, "r", encoding="utf-8") as f:
                        config.update(json.load(f))
                case "disable_multiprocessing":
                    config["pdftext_workers"] = 1
                case "disable_image_extraction":
                    config["extract_images"] = False
                case "use_llm":
                    # Skip use_llm from being added to config - it's only used for service selection
                    pass
                case _:
                    config[k] = v

        # Backward compatibility for google_api_key
        if settings.GOOGLE_API_KEY:
            config["gemini_api_key"] = settings.GOOGLE_API_KEY

        # Add LLM configuration from environment variables when use_llm is enabled
        if self.cli_options.get("use_llm", False):
            self._add_llm_config_from_env(config)

        return config

    def _add_llm_config_from_env(self, config: Dict[str, any]):
        """Add LLM configuration from environment variables to config dict."""
        # Gemini/Google
        if settings.GEMINI_API_KEY:
            config["gemini_api_key"] = settings.GEMINI_API_KEY
        if settings.GEMINI_MODEL_NAME:
            config["gemini_model_name"] = settings.GEMINI_MODEL_NAME
        if settings.THINKING_BUDGET:
            config["thinking_budget"] = settings.THINKING_BUDGET

        # OpenAI
        if settings.OPENAI_API_KEY:
            config["openai_api_key"] = settings.OPENAI_API_KEY
        if settings.OPENAI_MODEL:
            config["openai_model"] = settings.OPENAI_MODEL
        if settings.OPENAI_BASE_URL:
            config["openai_base_url"] = settings.OPENAI_BASE_URL
        if settings.OPENAI_IMAGE_FORMAT:
            config["openai_image_format"] = settings.OPENAI_IMAGE_FORMAT

        # Claude/Anthropic
        if settings.CLAUDE_API_KEY:
            config["claude_api_key"] = settings.CLAUDE_API_KEY
        if settings.CLAUDE_MODEL_NAME:
            config["claude_model_name"] = settings.CLAUDE_MODEL_NAME
        if settings.MAX_CLAUDE_TOKENS:
            config["max_claude_tokens"] = settings.MAX_CLAUDE_TOKENS

        # Azure OpenAI
        if settings.AZURE_ENDPOINT:
            config["azure_endpoint"] = settings.AZURE_ENDPOINT
        if settings.AZURE_API_KEY:
            config["azure_api_key"] = settings.AZURE_API_KEY
        if settings.AZURE_API_VERSION:
            config["azure_api_version"] = settings.AZURE_API_VERSION
        if settings.DEPLOYMENT_NAME:
            config["deployment_name"] = settings.DEPLOYMENT_NAME

        # Ollama
        if settings.OLLAMA_BASE_URL:
            config["ollama_base_url"] = settings.OLLAMA_BASE_URL
        if settings.OLLAMA_MODEL:
            config["ollama_model"] = settings.OLLAMA_MODEL

        # LLM Request Configuration
        if settings.TIMEOUT:
            config["timeout"] = settings.TIMEOUT
        if settings.MAX_RETRIES:
            config["max_retries"] = settings.MAX_RETRIES
        if settings.RETRY_WAIT_TIME:
            config["retry_wait_time"] = settings.RETRY_WAIT_TIME
        if settings.MAX_OUTPUT_TOKENS:
            config["max_output_tokens"] = settings.MAX_OUTPUT_TOKENS

    def get_llm_service(self):
        """
        Select the appropriate LLM service based on environment variables.

        Priority order:
        1. Explicitly set LLM_SERVICE environment variable
        2. Auto-detect based on which API key is available:
           - GEMINI_API_KEY → GoogleGeminiService
           - OPENAI_API_KEY → OpenAIService
           - CLAUDE_API_KEY → ClaudeService
           - AZURE_API_KEY → AzureOpenAIService
           - OLLAMA_BASE_URL → OllamaService
        3. Default to GoogleGeminiService with 'no-api-key' placeholder

        Only return a service when use_llm is enabled.
        """
        # Only return an LLM service when use_llm is enabled
        if not self.cli_options.get("use_llm", False):
            return None

        # Check for explicit LLM_SERVICE setting
        if settings.LLM_SERVICE:
            return settings.LLM_SERVICE

        # Auto-detect based on available API keys (priority order)
        if settings.GEMINI_API_KEY:
            return "marker.services.gemini.GoogleGeminiService"
        if settings.OPENAI_API_KEY:
            return "marker.services.openai.OpenAIService"
        if settings.CLAUDE_API_KEY:
            return "marker.services.claude.ClaudeService"
        if settings.AZURE_API_KEY:
            return "marker.services.azure.AzureOpenAIService"
        if settings.OLLAMA_BASE_URL:
            return "marker.services.ollama.OllamaService"

        # Default to Gemini service
        logger.warning(
            "No LLM API keys found in environment variables. "
            "Defaulting to GoogleGeminiService with placeholder key. "
            "Set one of: GEMINI_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY, "
            "AZURE_API_KEY, or OLLAMA_BASE_URL"
        )
        return "marker.services.gemini.GoogleGeminiService"

    def get_renderer(self):
        match self.cli_options["output_format"]:
            case "json":
                r = JSONRenderer
            case "markdown":
                r = MarkdownRenderer
            case "html":
                r = HTMLRenderer
            case "chunks":
                r = ChunkRenderer
            case _:
                raise ValueError("Invalid output format")
        return classes_to_strings([r])[0]

    def get_processors(self):
        processors = self.cli_options.get("processors", None)
        if processors is not None:
            processors = processors.split(",")
            for p in processors:
                try:
                    strings_to_classes([p])
                except Exception as e:
                    logger.error(f"Error loading processor: {p} with error: {e}")
                    raise

        return processors

    def get_converter_cls(self):
        converter_cls = self.cli_options.get("converter_cls", None)
        if converter_cls is not None:
            try:
                return strings_to_classes([converter_cls])[0]
            except Exception as e:
                logger.error(
                    f"Error loading converter: {converter_cls} with error: {e}"
                )
                raise

        return PdfConverter

    def get_output_folder(self, filepath: str):
        output_dir = self.cli_options.get("output_dir", settings.OUTPUT_DIR)
        fname_base = os.path.splitext(os.path.basename(filepath))[0]
        output_dir = os.path.join(output_dir, fname_base)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_base_filename(self, filepath: str):
        basename = os.path.basename(filepath)
        return os.path.splitext(basename)[0]
