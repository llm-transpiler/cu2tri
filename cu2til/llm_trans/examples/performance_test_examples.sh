#!/bin/bash

# Performance Testing Examples for cu2tri
# This script demonstrates various ways to use the performance testing system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 cu2tri Performance Testing Examples${NC}"
echo "=================================="

# Activate conda environment
echo -e "${YELLOW}📦 Activating conda environment 'serve'...${NC}"
conda activate serve

# Set base directory
BASE_DIR="/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler"
NVGPU_SERVER="http://localhost:8080"

echo -e "${BLUE}📂 Base Directory: $BASE_DIR${NC}"
echo -e "${BLUE}🖥️  NVGPU Server: $NVGPU_SERVER${NC}"
echo ""

# Function to run a command with nice formatting
run_example() {
    local description=$1
    local command=$2

    echo -e "${GREEN}➡️  $description${NC}"
    echo -e "${YELLOW}Command: $command${NC}"
    echo ""
    eval $command
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Example 1: List all successful kernels
run_example "List all successful kernels available for testing" \
    "python perf_cli.py list-kernels --base-dir $BASE_DIR"

# Example 2: Show statistics
run_example "Show statistics for all successful kernels" \
    "python perf_cli.py stats --base-dir $BASE_DIR"

# Example 3: Dry run for specific model
run_example "Dry run for gpt_5_mini model (show what would be tested)" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_5_mini --dry-run"

# Example 4: Test specific case types
run_example "Test only 'add' case types from gpt_5_mini model" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_5_mini --case-types add --warmup 5 --iters 50"

# Example 5: Test specific attempts
run_example "Test only attempt_01 from all models" \
    "python perf_cli.py test --base-dir $BASE_DIR --attempts 1 --warmup 3 --iters 20"

# Example 6: Test specific case with higher iterations
run_example "Test avgpool cases with more iterations for better accuracy" \
    "python perf_cli.py test --base-dir $BASE_DIR --case-types avgpool --warmup 20 --iters 200"

# Example 7: Concurrency testing
run_example "Run tests with concurrency of 2 (if server supports it)" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_5_mini --case-types add --concurrency 2 --warmup 5 --iters 50"

# Example 8: Full test for a specific recent run
run_example "Test all successful kernels from 20251023_013949 run" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_oss_120b --warmup 10 --iters 100"

# Example 9: Combined filters
run_example "Test add and avgpool cases from attempt_01 of gpt_5_mini" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_5_mini --case-types add --case-types avgpool --attempts 1 --warmup 15 --iters 150"

# Example 10: Performance with specific GPU
run_example "Test on specific GPU (e.g., GPU 0)" \
    "python perf_cli.py test --base-dir $BASE_DIR --model gpt_5_mini --case-types add --nvgpu-gpu 0 --warmup 10 --iters 100"

echo -e "${GREEN}✅ All examples completed!${NC}"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "  - Use --dry-run to see what would be tested without actually running tests"
echo "  - Increase --iters for more accurate timing (but longer runtime)"
echo "  - Use --concurrency to run multiple tests in parallel (if server supports it)"
echo "  - Check the generated logs in the original logs folders for detailed results"
echo "  - Make sure NVGPU server is running: cd /data/apps/project/cu2tri/server/nvgpu && python main.py"
echo ""
echo -e "${YELLOW}⚠️  Note: Performance tests run in exclusive GPU mode for accurate results${NC}"