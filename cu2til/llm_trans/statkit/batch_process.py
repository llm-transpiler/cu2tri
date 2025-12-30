#!/usr/bin/env python3

import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
import json


def discover_timestamps(runs_root: Path) -> Dict[str, List[Path]]:
    """Discover all timestamps under each model directory"""
    timestamps = defaultdict(list)

    # Look for model directories under runs/cu2tri/xpiler
    cu2tri_xpiler_root = runs_root / "cu2tri" / "xpiler"
    if not cu2tri_xpiler_root.exists():
        print(f"Error: {cu2tri_xpiler_root} does not exist")
        return timestamps

    for model_dir in cu2tri_xpiler_root.iterdir():
        if not model_dir.is_dir():
            continue

        # Look for timestamp directories
        for timestamp_dir in model_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            # Check if there's a xpiler.jsonl file in this directory
            jsonl_files = list(timestamp_dir.glob("*_xpiler.jsonl"))
            if jsonl_files:
                timestamps[model_dir.name].append(timestamp_dir)

    return timestamps


def get_processed_timestamps(stats_root: Path) -> Set[str]:
    """Get set of already processed timestamps from stats directory"""
    processed = set()

    stats_cu2tri_root = stats_root / "cu2tri"
    if not stats_cu2tri_root.exists():
        return processed

    for model_dir in stats_cu2tri_root.iterdir():
        if not model_dir.is_dir():
            continue

        for timestamp_dir in model_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            # Check if there's at least one output file
            if (timestamp_dir / "full_info.json").exists() or (timestamp_dir / "case_success.json").exists():
                processed.add(f"{model_dir.name}/{timestamp_dir.name}")

    return processed


def run_extraction_script(script_path: Path, jsonl_path: Path, output_dir: Path,
                          max_attempts: int = None, max_rounds: int = None,
                          overwrite: bool = False, minimal_only: bool = False) -> bool:
    """Run the extraction script on a specific JSONL file"""
    cmd = ["python3", str(script_path), str(jsonl_path)]

    if max_attempts is not None:
        cmd.extend(["--max-attempts", str(max_attempts)])
    if max_rounds is not None:
        cmd.extend(["--max-rounds", str(max_rounds)])
    if overwrite:
        cmd.append("--overwrite")
    if minimal_only:
        cmd.append("--minimal-only")
    cmd.extend(["--output-dir", str(output_dir)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully processed {jsonl_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {jsonl_path}: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch process all xpiler JSONL files and generate statistics")
    parser.add_argument("--runs-root", default="/data/apps/project/cu2tri/cu2til/llm_trans/runs",
                       help="Root directory for runs")
    parser.add_argument("--stats-root", default="/data/apps/project/cu2tri/cu2til/llm_trans/stats",
                       help="Root directory for stats output")
    parser.add_argument("--max-attempts", type=int, help="Maximum attempt number to include")
    parser.add_argument("--max-rounds", type=int, help="Maximum round number to include")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing stats files")
    parser.add_argument("--minimal-only", action="store_true", help="Generate only minimal case success stats")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without actually processing")
    parser.add_argument("--model", help="Process only specific model (e.g., 'gpt_5_mini')")
    parser.add_argument("--timestamp", help="Process only specific timestamp (e.g., '20251024_065400')")

    args = parser.parse_args()

    # Paths
    runs_root = Path(args.runs_root)
    stats_root = Path(args.stats_root)
    script_path = Path(__file__).parent / "extract_stats.py"

    if not script_path.exists():
        print(f"Error: Extraction script {script_path} does not exist")
        return 1

    # Discover available timestamps
    print("Discovering available timestamps...")
    timestamps = discover_timestamps(runs_root)

    if not timestamps:
        print("No timestamp directories found with xpiler JSONL files")
        return 1

    # Filter by model if specified
    if args.model:
        if args.model not in timestamps:
            print(f"Error: Model '{args.model}' not found. Available models: {list(timestamps.keys())}")
            return 1
        timestamps = {args.model: timestamps[args.model]}

    # Get already processed timestamps
    processed = get_processed_timestamps(stats_root)

    # Collect files to process
    files_to_process = []
    total_files = 0
    skipped_files = 0

    for model_name, timestamp_dirs in timestamps.items():
        print(f"\nModel: {model_name}")

        for timestamp_dir in timestamp_dirs:
            timestamp_name = timestamp_dir.name
            identifier = f"{model_name}/{timestamp_name}"

            # Skip if specific timestamp requested and this is not it
            if args.timestamp and timestamp_name != args.timestamp:
                continue

            # Find JSONL files
            jsonl_files = list(timestamp_dir.glob("*_xpiler.jsonl"))
            if not jsonl_files:
                continue

            total_files += len(jsonl_files)

            # Check if already processed
            if identifier in processed and not args.overwrite:
                print(f"  ⏭  {timestamp_name}: already processed ({len(jsonl_files)} files)")
                skipped_files += len(jsonl_files)
                continue

            print(f"  📋 {timestamp_name}: {len(jsonl_files)} files to process")
            for jsonl_file in jsonl_files:
                files_to_process.append((jsonl_file, identifier))

    print(f"\nSummary:")
    print(f"  Total JSONL files found: {total_files}")
    print(f"  Files to process: {len(files_to_process)}")
    print(f"  Files to skip: {skipped_files}")

    if args.dry_run:
        print("\nDry run - not processing any files")
        return 0

    if not files_to_process:
        print("No files to process")
        return 0

    # Process files
    print(f"\nProcessing {len(files_to_process)} files...")

    success_count = 0
    failure_count = 0

    for i, (jsonl_file, identifier) in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing {identifier}: {jsonl_file.name}")

        if run_extraction_script(
            script_path, jsonl_file, stats_root,
            args.max_attempts, args.max_rounds, args.overwrite, args.minimal_only
        ):
            success_count += 1
        else:
            failure_count += 1

    print(f"\nBatch processing completed!")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {failure_count}")
    print(f"Skipped: {skipped_files}")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    exit(main())