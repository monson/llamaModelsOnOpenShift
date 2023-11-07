#!/bin/bash
# Stop on any error
set -e

# If the HF_TOKEN variable is set, log in to the Hugging Face CLI
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face with the provided token."
    huggingface-cli login --token $HF_TOKEN
else
    echo "No Hugging Face token provided. Proceeding without logging in."
fi

# Execute the main container command (CMD in Dockerfile)
exec "$@"
