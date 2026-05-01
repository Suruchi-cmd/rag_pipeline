#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

export OLLAMA_KEEP_ALIVE=-1

# Build frontend if dist is missing
if [ ! -d "frontend/dist" ]; then
    echo "Building frontend..."
    cd frontend && npm run build-only && cd ..
fi

echo "Starting AeroBot on http://localhost:3232 ..."
python main.py
