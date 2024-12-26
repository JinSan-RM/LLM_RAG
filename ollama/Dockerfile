# Use Ollama base image
FROM ollama/ollama:latest

# Set environment variable for Ollama model path
ENV OLLAMA_MODEL_PATH=/models

# Set working directory to /app
WORKDIR /app

# Expose port 11434 for Ollama API
EXPOSE 11434

# Install necessary packages
RUN apt-get update && apt-get install -y curl dos2unix && rm -rf /var/lib/apt/lists/*
# Create volume to persist models
VOLUME ["/models"]

# Copy entrypoint script and set execute permissions
COPY llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf /usr/local/bin/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf
COPY Bllossom_Modelfile /usr/local/bin/Modelfile

COPY EEVE_Modelfile /usr/local/bin/EEVE_Modelfile
COPY EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf /usr/local/bin/EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/Modelfile && chmod +x /usr/local/bin/Modelfile

# Use entrypoint.sh as the entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
