#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3232}"

echo "Starting AeroBot voice server..."
echo "Host: $HOST | Port: $PORT"

export OLLAMA_KEEP_ALIVE=-1  # never unload
uvicorn chatbot.server:app --host "$HOST" --port "$PORT"
