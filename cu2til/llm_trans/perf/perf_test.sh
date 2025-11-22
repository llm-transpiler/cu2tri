#!/bin/bash

# Performance Testing Script for Successful Triton Kernels

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to show usage
show_usage() {
    echo "Performance Testing for Successful Triton Kernels"
    echo "Usage: $0 [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  all                           Test all successful kernels"
    echo "  model MODEL                   Test kernels from specific model only"
    echo "  help                          Show this help message"
    echo ""
    echo "Options:"
    echo "  --gpu ID                      GPU ID for exclusive mode (default: 7)"
    echo "  --timestamp TS                Test only specific timestamp"
    echo "  --case-type TYPE              Test only specific case type (e.g., 'add' for all add cases)"
    echo "  --case-name CASE              Test only specific case name (e.g., 'add_1_15_64')"
    echo "  --dry-run                     Show what would be tested without running tests"
    echo ""
    echo "Examples:"
    echo "  $0 all                                                   # Test all successful kernels on GPU 7"
    echo "  $0 all --gpu 3                                         # Test all on GPU 3"
    echo "  $0 model gpt_oss_120b                                  # Test only gpt_oss_120b model"
    echo "  $0 model gpt_5_mini --gpu 5                            # Test gpt_5_mini on GPU 5"
    echo "  $0 all --dry-run                                        # Show what would be tested"
    echo "  $0 model gpt_oss_120b --timestamp 20251122_054219     # Test specific model/timestamp"
    echo "  $0 all --case-type add                                # Test all add cases across all models"
    echo "  $0 all --case-name add_1_15_64                        # Test specific add case"
    echo "  $0 model gpt_oss_120b --case-type gemm                # Test all gemm cases for specific model"
    echo "  $0 model gpt_5_mini --case-type add --dry-run         # Dry run for specific model/case-type"
}

# Parse command line arguments
COMMAND=""
GPU=""
MODEL=""
TIMESTAMP=""
CASE_TYPE=""
CASE_NAME=""
DRY_RUN=""

# Extract command first
if [[ $# -eq 0 ]]; then
    COMMAND="all"
else
    case $1 in
        all|help)
            COMMAND="$1"
            shift
            ;;
        model)
            COMMAND="model"
            if [[ $# -lt 2 ]]; then
                echo "Error: 'model' command requires a model name"
                show_usage
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        dry-run)
            COMMAND="all"
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --gpu)
            if [[ -z "$2" ]]; then
                echo "Error: --gpu requires a GPU ID"
                show_usage
                exit 1
            fi
            GPU="--gpu $2"
            shift 2
            ;;
        --timestamp)
            if [[ -z "$2" ]]; then
                echo "Error: --timestamp requires a timestamp"
                show_usage
                exit 1
            fi
            TIMESTAMP="--timestamp $2"
            shift 2
            ;;
        --case-type)
            if [[ -z "$2" ]]; then
                echo "Error: --case-type requires a case type"
                show_usage
                exit 1
            fi
            CASE_TYPE="--case-type $2"
            shift 2
            ;;
        --case-name)
            if [[ -z "$2" ]]; then
                echo "Error: --case-name requires a case name"
                show_usage
                exit 1
            fi
            CASE_NAME="--case-name $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute commands
case $COMMAND in
    all)
        echo "Testing all successful triton kernels..."
        python3 "$SCRIPT_DIR/run_performance_tests.py" $DRY_RUN $GPU $TIMESTAMP $CASE_TYPE $CASE_NAME
        ;;
    model)
        echo "Testing kernels for model: $MODEL"
        python3 "$SCRIPT_DIR/run_performance_tests.py" --model "$MODEL" $DRY_RUN $GPU $TIMESTAMP $CASE_TYPE $CASE_NAME
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac