#!/bin/bash
# Start NPU Server

set -e

PROJECT_ROOT=${PROJECT_ROOT:-/workspace}
HOST=${NPU_HOST:-0.0.0.0}
PORT=${NPU_PORT:-8080}
NPU_CONFIG=${NPU_CONFIG:-server/npu/configs/npu_resources/ascend_910b_sample.yml}

cd "$PROJECT_ROOT"

echo "Starting NPU Server on $HOST:$PORT"
echo "NPU Config: $NPU_CONFIG"
echo ""

python -m server.npu.main \
    --host "$HOST" \
    --port "$PORT" \
    --npu-config "$NPU_CONFIG" \
    "$@"
#!/bin/bash
# Start NPU Server

set -e

PROJECT_ROOT=${PROJECT_ROOT:-/workspace}
HOST=${NPU_HOST:-0.0.0.0}
PORT=${NPU_PORT:-8080}
NPU_CONFIG=${NPU_CONFIG:-server/npu/configs/npu_resources/ascend_910b_sample.yml}

cd "$PROJECT_ROOT"

echo "Starting NPU Server on $HOST:$PORT"
echo "NPU Config: $NPU_CONFIG"
echo ""

python -m server.npu.main \
    --host "$HOST" \
    --port "$PORT" \
    --npu-config "$NPU_CONFIG" \
    "$@"
