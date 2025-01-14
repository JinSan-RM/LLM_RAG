#!/bin/bash

echo "Starting entrypoint.sh script..." > /tmp/entrypoint_debug.log
echo "Starting Ollama server..." >> /tmp/entrypoint_debug.log

ollama serve &  # &는 백그라운드 실행을 의미
server_pid=$!   # Ollama 서버 프로세스 ID 저장

echo "Downloading models..." >> /tmp/entrypoint_debug.log
ollama pull llama3.2 >> /tmp/entrypoint_debug.log 2>&1
ollama pull nomic-embed-text >> /tmp/entrypoint_debug.log 2>&1

echo "Entrypoint script completed." >> /tmp/entrypoint_debug.log



# ollama create bllossom -f /usr/local/bin/Modelfile
# echo "ollama bllossom creating." >> /tmp/entrypoint_debug.log

ollama create solar -f /usr/local/bin/SolarModelfile
ollama create EEVE -f /usr/local/bin/EEVEModelfile


wait $server_pid