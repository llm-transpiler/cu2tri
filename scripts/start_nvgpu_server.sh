#!/bin/bash
# Start NVGPU Server

set -e

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/cu2tri}
HOST=${NVGPU_HOST:-0.0.0.0}
PORT=${NVGPU_PORT:-8080}
GPU_CONFIG=${GPU_CONFIG:-nvgpu_server/nvgpu/configs/gpu_resources/P250_A6000.yml}

# Change to project root
cd "$PROJECT_ROOT"

# Install if needed
if ! command -v nvgpu-server &> /dev/null; then
    echo "Installing nvgpu-server..."
    pip install -e ./nvgpu_server
fi

# Start server
echo "Starting NVGPU Server on $HOST:$PORT"
echo "GPU Config: $GPU_CONFIG"
echo ""

nvgpu-server \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-config "$GPU_CONFIG" \
    "$@"
