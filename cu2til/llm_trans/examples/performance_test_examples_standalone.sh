#!/bin/bash
# Performance Testing Examples for cu2tri (Standalone Version)
# This script demonstrates various ways to use the standalone performance testing system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 cu2tri Standalone Performance Testing Examples${NC}"
echo "================================================="

# Set base directory
BASE_DIR="/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400"
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
    "python perf_cli_standalone.py list-kernels --base-dir $BASE_DIR | head -20"

# Example 2: Show statistics
run_example "Show statistics for all successful kernels" \
    "python perf_cli_standalone.py stats --base-dir $BASE_DIR"

# Example 3: Filter by case types
run_example "Show statistics for 'add' case types only" \
    "python perf_cli_standalone.py stats --base-dir $BASE_DIR --case-types add"

# Example 4: List specific case types and attempts
run_example "List only 'add' cases from attempt_01" \
    "python perf_cli_standalone.py list-kernels --base-dir $BASE_DIR --case-types add --attempts 1"

# Example 5: Dry run for specific case types
run_example "Dry run for 'add' case types from attempt_01 (show what would be tested)" \
    "python perf_cli_standalone.py test --base-dir $BASE_DIR --case-types add --attempts 1 --warmup 5 --iters 20 --dry-run"

# Example 6: List different case types
run_example "List 'avgpool' and 'gelu' case types" \
    "python perf_cli_standalone.py list-kernels --base-dir $BASE_DIR --case-types avgpool --case-types gelu | head -15"

# Example 7: Show statistics for multiple case types
run_example "Show statistics for 'add', 'avgpool', and 'gelu' case types" \
    "python perf_cli_standalone.py stats --base-dir $BASE_DIR --case-types add --case-types avgpool --case-types gelu"

# Example 8: Actual performance test (if NVGPU server is available)
echo -e "${YELLOW}⚠️  Note: The following example requires NVGPU server to be running${NC}"
echo -e "${YELLOW}    Start server with: cd /data/apps/project/cu2tri/server/nvgpu && python main.py${NC}"
echo ""

run_example "Run actual performance test on 'add_1_15_64' (attempt_01 only)" \
    "python perf_cli_standalone.py test --base-dir $BASE_DIR --case-types add --attempts 1 --warmup 3 --iters 10 --nvgpu-server $NVGPU_SERVER || echo 'NVGPU server not available - skipping actual performance test'"

echo -e "${GREEN}✅ All examples completed!${NC}"
echo ""
echo -e "${BLUE}💡 Usage Tips:${NC}"
echo "  - Use --dry-run to see what would be tested without actually running tests"
echo "  - Use --case-types to filter by specific operation types (add, avgpool, gelu, etc.)"
echo "  - Use --attempts to test only specific attempts (1 for attempt_01, 2 for attempt_02, etc.)"
echo "  - Adjust --warmup and --iters for more/less accurate timing"
echo "  - The standalone version doesn't need complex cu2tri dependencies"
echo "  - Performance logs are saved to the original logs/ folders"
echo "  - Make sure NVGPU server is running for actual performance tests"
echo ""
echo -e "${BLUE}🔍 Available Commands:${NC}"
echo "  list-kernels  - List successful kernels matching criteria"
echo "  stats         - Show statistics about successful kernels"
echo "  test          - Run performance tests (dry-run or actual)"
echo ""
echo -e "${YELLOW}🖥️  NVGPU Server:${NC}"
echo "  cd /data/apps/project/cu2tri/server/nvgpu && python main.py"
echo ""
echo -e "${GREEN}🎉 Performance testing system is ready to use!${NC}"