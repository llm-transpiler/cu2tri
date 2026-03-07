#!/bin/bash
# Start LLM Transpiler

set -e

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/cu2tri}
MODEL=${LLM_MODEL:-gpt_oss_120b}
TESTSET=${LLM_TESTSET:-xpiler}

# Change to project root
cd "$PROJECT_ROOT"

# Install if needed
if ! command -v llm-trans &> /dev/null; then
    echo "Installing llm-call..."
    pip install -e ./llm_call
fi

# Run transpiler
echo "Starting LLM Transpiler"
echo "Model: $MODEL"
echo "Test Set: $TESTSET"
echo ""

llm-trans \
    --model "$MODEL" \
    --testset "$TESTSET" \
    "$@"
