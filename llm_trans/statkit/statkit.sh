#!/bin/bash

# CU2TRI Statistics Toolkit - Quick Start Scripts

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to show usage
show_usage() {
    echo "CU2TRI Statistics Toolkit - Quick Commands"
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  all               Process all JSONL files in runs/cu2tri"
    echo "  single <path>     Process a single file or directory"
    echo "  dry-run           Show what would be processed without actually processing"
    echo "  interactive       Interactive mode (default)"
    echo "  help              Show this help message"
    echo ""
    echo "Options (can be combined with commands):"
    echo "  --max-attempts N  Limit to N attempts per case"
    echo "  --max-rounds N    Limit to N rounds per case"
    echo "  --overwrite       Overwrite existing files"
    echo "  --minimal-only    Generate only minimal case success stats"
    echo "  --model MODEL     Process only specific model"
    echo "  --timestamp TS    Process only specific timestamp"
    echo ""
    echo "Examples:"
    echo "  $0 all                           # Process all files"
    echo "  $0 all --max-attempts 3          # All files, max 3 attempts"
    echo "  $0 dry-run --model gpt_5_mini    # Dry run for specific model"
    echo "  $0 single /path/to/file.jsonl    # Process single file"
}

# Default command
COMMAND=${1:-interactive}
shift

# Parse common options
MAX_ATTEMPTS=""
MAX_ROUNDS=""
OVERWRITE=""
MINIMAL_ONLY=""
MODEL=""
TIMESTAMP=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-attempts)
            MAX_ATTEMPTS="--max-attempts $2"
            shift 2
            ;;
        --max-rounds)
            MAX_ROUNDS="--max-rounds $2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        --minimal-only)
            MINIMAL_ONLY="--minimal-only"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --timestamp)
            TIMESTAMP="$2"
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
        echo "Processing all JSONL files..."
        python3 "$SCRIPT_DIR/batch_process.py" $MAX_ATTEMPTS $MAX_ROUNDS $OVERWRITE $MINIMAL_ONLY $MODEL $TIMESTAMP
        ;;
    single)
        if [[ -z "$1" ]]; then
            echo "Error: Please provide a path to a JSONL file or directory"
            show_usage
            exit 1
        fi
        echo "Processing: $1"
        python3 "$SCRIPT_DIR/extract_stats.py" "$1" $MAX_ATTEMPTS $MAX_ROUNDS $OVERWRITE $MINIMAL_ONLY
        ;;
    dry-run)
        DRY_RUN="--dry-run"
        echo "Dry run - showing what would be processed..."
        python3 "$SCRIPT_DIR/batch_process.py" $DRY_RUN $MAX_ATTEMPTS $MAX_ROUNDS $MODEL $TIMESTAMP
        ;;
    interactive)
        python3 "$SCRIPT_DIR/statkit.py"
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