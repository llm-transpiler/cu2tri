#!/bin/bash
# NVGPU Server Test Runner Script

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           NVGPU Server Test Suite Runner                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: Please run this script from the /workspace/server/nvgpu directory${NC}"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="all"
VERBOSE=""
COVERAGE=true
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            MARKERS="-m unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            MARKERS="-m integration"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v -s"
            shift
            ;;
        --file)
            TEST_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit            Run only unit tests"
            echo "  --integration     Run only integration tests"
            echo "  --no-coverage     Skip coverage report"
            echo "  -v, --verbose     Verbose output"
            echo "  --file <file>     Run specific test file"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all tests with coverage"
            echo "  $0 --unit                   # Run only unit tests"
            echo "  $0 --file test_models.py    # Run specific test file"
            echo "  $0 -v --no-coverage         # Verbose without coverage"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if test dependencies are installed
echo -e "${YELLOW}🔍 Checking test dependencies...${NC}"
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  pytest not found. Installing test dependencies...${NC}"
    pip install -r tests/requirements-test.txt
    echo -e "${GREEN}✅ Test dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Test dependencies found${NC}"
fi
echo ""

# Build pytest command
PYTEST_CMD="pytest tests/"

if [ -n "$TEST_FILE" ]; then
    PYTEST_CMD="pytest tests/$TEST_FILE"
    echo -e "${YELLOW}📝 Running tests from: $TEST_FILE${NC}"
else
    echo -e "${YELLOW}📝 Running $TEST_TYPE tests${NC}"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=term-missing --cov-report=html"
    echo -e "${YELLOW}📊 Coverage enabled${NC}"
fi

if [ -n "$VERBOSE" ]; then
    PYTEST_CMD="$PYTEST_CMD $VERBOSE"
    echo -e "${YELLOW}🔊 Verbose mode enabled${NC}"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Run the tests
if eval $PYTEST_CMD; then
    EXIT_CODE=0
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}📊 Coverage report generated:${NC}"
        echo -e "   HTML: ${YELLOW}htmlcov/index.html${NC}"
        echo -e "   Command to view: ${YELLOW}open htmlcov/index.html${NC} (macOS) or ${YELLOW}xdg-open htmlcov/index.html${NC} (Linux)"
    fi
else
    EXIT_CODE=1
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo -e "   • Run with ${YELLOW}-v${NC} for more details"
    echo -e "   • Use ${YELLOW}--pdb${NC} to debug failed tests"
    echo -e "   • Run specific test: ${YELLOW}$0 --file test_models.py${NC}"
fi

echo ""
exit $EXIT_CODE

