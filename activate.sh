#!/bin/bash
# OpenEvolve Environment Activation Script
# Usage: source activate.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Load .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "Loaded environment variables from .env"
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "WARNING: OPENAI_API_KEY is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-gemini-api-key'"
    echo "Get your key from: https://aistudio.google.com/apikey"
else
    echo "OPENAI_API_KEY is configured (length: ${#OPENAI_API_KEY})"
fi

echo ""
echo "OpenEvolve environment activated!"
echo "Run 'openevolve-run --help' for usage information"
