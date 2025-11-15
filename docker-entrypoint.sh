#!/bin/bash
set -e

echo "======================================"
echo "Marker PDF Server - Docker Container"
echo "======================================"

# Pre-download models if they don't exist
if [ ! -d "/root/.cache/huggingface/hub" ] || [ -z "$(ls -A /root/.cache/huggingface/hub 2>/dev/null)" ]; then
    echo "First run detected - downloading models..."
    echo "This may take several minutes depending on your internet connection."
    python3 -c "from marker.models import create_model_dict; create_model_dict()" || {
        echo "Warning: Model download may have failed, but continuing anyway..."
    }
    echo "Models downloaded successfully!"
else
    echo "Models already cached, skipping download."
fi

echo ""
echo "Starting Marker PDF Server..."
echo "Device: ${TORCH_DEVICE}"
echo "Port: ${PORT:-8001}"
echo "Host: ${HOST:-0.0.0.0}"

# LLM Configuration Detection (based on configured API keys)
if [ -n "$GEMINI_API_KEY" ]; then
    echo "LLM Provider Available: Gemini (Google)"
elif [ -n "$OPENAI_API_KEY" ]; then
    echo "LLM Provider Available: OpenAI"
elif [ -n "$CLAUDE_API_KEY" ]; then
    echo "LLM Provider Available: Claude (Anthropic)"
elif [ -n "$AZURE_API_KEY" ]; then
    echo "LLM Provider Available: Azure OpenAI"
elif [ -n "$OLLAMA_BASE_URL" ]; then
    echo "LLM Provider Available: Ollama (Local)"
else
    echo "LLM Provider: None configured (set API keys to enable)"
fi
echo "Note: Use OpenWebUI toggle to enable/disable LLM per request"

echo ""
echo "Server will be available at: http://${HOST:-0.0.0.0}:${PORT:-8001}"
echo "API Documentation: http://${HOST:-0.0.0.0}:${PORT:-8001}/docs"
echo ""
echo "======================================"
echo ""

# Start marker_server with basic configuration
# All LLM configuration is read directly from environment variables by the Python code
exec marker_server --port ${PORT:-8001} --host ${HOST:-0.0.0.0}
