#!/bin/bash
# Quick test script for Triton→CUTE translation

set -e  # Exit on error

# Configuration
CONDA_ENV="serve"
CUTLASS_ROOT="${CUTLASS_ROOT:-/data/apps/project/cu2tri/elib/cutlass_latest}"
GPU_ID="${GPU_ID:-7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Triton→CUTE Translation Test${NC}"
echo -e "${GREEN}========================================${NC}"

# Check conda environment
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}Error: Conda environment '${CONDA_ENV}' not found${NC}"
    exit 1
fi

# Export environment variables
export CUTLASS_ROOT
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo -e "${YELLOW}Environment:${NC}"
echo "  CONDA_ENV: $CONDA_ENV"
echo "  CUTLASS_ROOT: $CUTLASS_ROOT"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Test selection
TESTSET="${1:-triton2cute_add}"
MODEL="${2:-gpt_5_mini}"
MAX_ROUNDS="${3:-3}"

echo -e "${YELLOW}Test Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Testset: $TESTSET"
echo "  Max rounds: $MAX_ROUNDS"
echo ""

# Run test
echo -e "${GREEN}Starting test...${NC}"
conda run -n $CONDA_ENV python -m cu2til.llm_trans \
  --model $MODEL \
  --testset $TESTSET \
  --source-lang triton \
  --target-lang cute \
  --max-rounds $MAX_ROUNDS \
  --max-attempts 1 \
  --concurrency 1 \
  --no-nvgpu

# Check result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Test failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

