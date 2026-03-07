#!/usr/bin/env python3

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re


def extract_case_info(case_entry: dict) -> dict:
    """Extract relevant information from a case entry"""
    return {
        "case_type": case_entry.get("case_type"),
        "case_name": case_entry.get("case_name"),
        "attempt_number": case_entry.get("attempt_number"),
        "success": case_entry.get("success", False),
        "final_round": case_entry.get("final_round"),
        "start_time": case_entry.get("start_time"),
        "end_time": case_entry.get("end_time"),
        "wall_clock_duration_ms": case_entry.get("wall_clock_duration_ms"),
        "effective_duration_ms": case_entry.get("effective_duration_ms"),
        "total_llm_time_ms": case_entry.get("total_llm_time_ms"),
        "total_test_time_ms": case_entry.get("total_test_time_ms"),
        "work_dir": case_entry.get("work_dir"),
        "aggregated_timers": case_entry.get("aggregated_timers", {}),
        "llm_rounds": case_entry.get("llm_rounds", []),
        "test_rounds": case_entry.get("test_rounds", [])
    }


def natural_sort_key(text: str) -> list:
    """Natural sorting key that handles numbers properly"""
    return [int(c) if c.isdigit() else c for c in re.split(r'([0-9]+)', text)]


def filter_attempts_by_max(attempts: List[dict], max_attempts: Optional[int]) -> List[dict]:
    """Filter attempts to only include those with attempt_number <= max_attempts"""
    if max_attempts is None:
        return attempts
    return [attempt for attempt in attempts if attempt["attempt_number"] <= max_attempts]


def filter_rounds_by_max(case_data: dict, max_rounds: Optional[int]) -> dict:
    """Filter test rounds to only include those with round number <= max_rounds"""
    if max_rounds is None:
        return case_data

    filtered_case_data = case_data.copy()
    if "test_rounds" in filtered_case_data:
        filtered_case_data["test_rounds"] = [
            round_data for round_data in filtered_case_data["test_rounds"]
            if round_data.get("round", 1) <= max_rounds
        ]

    if "llm_rounds" in filtered_case_data:
        for llm_round in filtered_case_data["llm_rounds"]:
            if "retry_records" in llm_round:
                llm_round["retry_records"] = [
                    retry for retry in llm_round["retry_records"]
                    if retry.get("retry_index", 0) + 1 <= max_rounds
                ]

    return filtered_case_data


def process_jsonl_file(jsonl_path: Path, max_attempts: Optional[int], max_rounds: Optional[int]) -> Tuple[dict, List[dict]]:
    """Process a single JSONL file and extract metadata and case information"""
    metadata = None
    cases = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)

                if entry.get("_type") == "metadata":
                    metadata = entry
                elif "case_type" in entry and "case_name" in entry:
                    case_info = extract_case_info(entry)
                    case_info = filter_rounds_by_max(case_info, max_rounds)
                    cases.append(case_info)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line in {jsonl_path}: {e}")
                continue

    return metadata, cases


def organize_cases(cases: List[dict]) -> Dict[str, List[dict]]:
    """Organize cases by case_name and sort attempts by attempt_number"""
    organized = defaultdict(list)

    for case in cases:
        case_name = case["case_name"]
        organized[case_name].append(case)

    # Sort attempts within each case by attempt_number
    for case_name in organized:
        organized[case_name].sort(key=lambda x: x["attempt_number"])

    return organized


def generate_full_stats_json(metadata: dict, organized_cases: Dict[str, List[dict]], max_attempts: Optional[int], max_rounds: Optional[int]) -> dict:
    """Generate the full stats JSON structure"""
    result = {
        "metadata": metadata.copy() if metadata else {},
        "summary": {
            "total_cases": len(organized_cases),
            "case_types": list(set(case["case_type"] for case_list in organized_cases.values() for case in case_list)),
            "total_attempts": sum(len(case_list) for case_list in organized_cases.values())
        },
        "cases": {}
    }

    # Add filter information to metadata
    if max_attempts is not None:
        result["metadata"]["max_attempts_filter"] = max_attempts
    if max_rounds is not None:
        result["metadata"]["max_rounds_filter"] = max_rounds

    # Sort case names naturally
    sorted_case_names = sorted(organized_cases.keys(), key=natural_sort_key)

    for case_name in sorted_case_names:
        attempts = organized_cases[case_name]
        successful_attempts = [attempt for attempt in attempts if attempt.get("success", False)]

        result["cases"][case_name] = {
            "case_type": attempts[0]["case_type"] if attempts else None,
            "total_attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "attempts": attempts
        }

    return result


def generate_config_json(metadata: dict, organized_cases: Dict[str, List[dict]]) -> dict:
    """Generate config.json with original data information (actual attempts and rounds)"""

    # Calculate original data statistics
    total_cases = len(organized_cases)
    case_types = set()
    max_attempts_per_case = {}
    max_rounds_per_case = {}

    for case_name, attempts in organized_cases.items():
        if not attempts:
            continue

        case_type = attempts[0]["case_type"]
        case_types.add(case_type)

        # Track actual attempts and rounds for this case
        actual_attempts = len(attempts)
        max_rounds = 0

        for attempt in attempts:
            if attempt.get("test_rounds"):
                round_count = len(attempt.get("test_rounds", []))
                max_rounds = max(max_rounds, round_count)

        max_attempts_per_case[case_name] = actual_attempts
        max_rounds_per_case[case_name] = max_rounds

    # Configuration metadata with original data info
    config_json = {
        "model_name": metadata.get("model_name") if metadata else None,
        "timestamp": metadata.get("timestamp") if metadata else None,
        "case_base_info": {
            "total_cases": total_cases,
            "case_types": sorted(list(case_types))
        }
    }

    return config_json


def generate_minimal_stats_json(metadata: dict, organized_cases: Dict[str, List[dict]], max_attempts: Optional[int], max_rounds: Optional[int]) -> dict:
    """Generate minimal stats JSON structure with case_type as top-level keys and detailed attempt info"""

    # Case statistics organized by case_type
    case_stats = {}

    # Sort case names naturally
    sorted_case_names = sorted(organized_cases.keys(), key=natural_sort_key)

    for case_name in sorted_case_names:
        attempts = organized_cases[case_name]
        if not attempts:
            continue

        # Extract case_type from first attempt
        case_type = attempts[0]["case_type"]

        # Ensure case_type exists in case_stats
        if case_type not in case_stats:
            case_stats[case_type] = {}

        # Prepare attempt statistics
        attempt_stats = []
        final_success = False

        # Sort attempts by attempt_number
        attempts.sort(key=lambda x: x["attempt_number"])

        for attempt in attempts:
            # Apply max_attempts filter if specified
            if max_attempts is not None and attempt["attempt_number"] > max_attempts:
                continue

            attempt_stat = {
                "attempt_number": attempt["attempt_number"],
                "success": attempt.get("success", False),
                "round_final": None
            }

            if attempt.get("success", False):
                # Get the successful round number as integer
                if attempt.get("test_rounds"):
                    successful_rounds = [r.get("round", 1) for r in attempt.get("test_rounds", []) if r.get("success", False)]
                    if successful_rounds:
                        attempt_stat["round_final"] = successful_rounds[0]  # First successful round as integer
                        final_success = True
                    else:
                        attempt_stat["round_final"] = None
                else:
                    attempt_stat["round_final"] = None
            else:
                # For failed attempts, get the last round attempted
                if attempt.get("test_rounds"):
                    max_round = max(r.get("round", 1) for r in attempt.get("test_rounds", []))
                    attempt_stat["round_final"] = max_round
                else:
                    attempt_stat["round_final"] = 1  # Default to round 1 if no test rounds found

            attempt_stats.append(attempt_stat)

        # Add to case_stats
        case_stats[case_type][case_name] = {
            "stat_final_success": final_success,
            "stat_total_attempts": len(attempt_stats),
            "attempts": attempt_stats
        }

    return case_stats


def generate_filename(base_name: str, max_attempts: Optional[int], max_rounds: Optional[int], is_full: bool = True) -> str:
    """Generate output filename with appropriate suffixes"""
    if is_full:
        # Full version - always use full_info.json
        suffixes = []
        if max_attempts is not None:
            suffixes.append(f"at{max_attempts}")
        if max_rounds is not None:
            suffixes.append(f"r{max_rounds}")

        if suffixes:
            return f"full_info_{'_'.join(suffixes)}.json"
        else:
            return "full_info.json"
    else:
        # Minimal version - use concise naming
        suffixes = []
        if max_attempts is not None:
            suffixes.append(f"at{max_attempts}")
        if max_rounds is not None:
            suffixes.append(f"r{max_rounds}")

        if suffixes:
            return f"case_success_{'_'.join(suffixes)}.json"
        else:
            return "case_success.json"


def main():
    parser = argparse.ArgumentParser(description="Extract statistics from xpiler JSONL files")
    parser.add_argument("jsonl_path", nargs='?', default="/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri",
                       help="Path to the JSONL file or directory containing JSONL files (default: runs/cu2tri)")
    parser.add_argument("--max-attempts", type=int, help="Maximum attempt number to include")
    parser.add_argument("--max-rounds", type=int, help="Maximum round number to include")
    parser.add_argument("--output-dir", default="/data/apps/project/cu2tri/cu2til/llm_trans/stats",
                       help="Output directory for stats files")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing stats files (default: skip existing files)")
    parser.add_argument("--minimal-only", action="store_true",
                       help="Generate only minimal case success stats")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(args.jsonl_path)

    # Handle both single files and directories
    if jsonl_path.is_file():
        jsonl_files = [jsonl_path]
    elif jsonl_path.is_dir():
        # Find all *_xpiler.jsonl files recursively
        jsonl_files = list(jsonl_path.rglob("*_xpiler.jsonl"))
        print(f"Found {len(jsonl_files)} JSONL files to process")
    else:
        print(f"Error: Path {jsonl_path} does not exist")
        return 1

    processed_count = 0
    skipped_count = 0

    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file}...")

        # Determine output path maintaining directory structure
        try:
            rel_path = jsonl_file.relative_to(Path("/data/apps/project/cu2tri/cu2til/llm_trans/runs"))
        except ValueError:
            # If file is not under runs, use its relative path from the file itself
            rel_path = jsonl_file.parent / jsonl_file.name

        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Generate filenames
        output_files = []
        if not args.minimal_only:
            full_filename = generate_filename("full_info", args.max_attempts, args.max_rounds, is_full=True)
            output_files.append(("full", full_filename))

        minimal_filename = generate_filename("case_success", args.max_attempts, args.max_rounds, is_full=False)
        output_files.append(("minimal", minimal_filename))

        # Check if any files need to be generated
        skip_all = True
        for file_type, filename in output_files:
            output_path = output_subdir / filename
            if not output_path.exists() or args.overwrite:
                skip_all = False
                break

        if skip_all:
            print(f"Skipping {jsonl_file} - output files already exist (use --overwrite to regenerate)")
            skipped_count += 1
            continue

        # Extract metadata and cases
        metadata, cases = process_jsonl_file(jsonl_file, args.max_attempts, args.max_rounds)

        if not metadata:
            print(f"Warning: No metadata found in {jsonl_file}")
            continue

        if not cases:
            print(f"Warning: No cases found in {jsonl_file}")
            continue

        # Filter attempts if specified
        if args.max_attempts is not None:
            # Group cases by case_name first, then filter attempts
            case_groups = defaultdict(list)
            for case in cases:
                case_groups[case["case_name"]].append(case)

            filtered_cases = []
            for case_name, case_attempts in case_groups.items():
                filtered_attempts = filter_attempts_by_max(case_attempts, args.max_attempts)
                filtered_cases.extend(filtered_attempts)
            cases = filtered_cases

        # Organize cases
        organized_cases = organize_cases(cases)

        # Generate and write stats files
        for file_type, filename in output_files:
            output_path = output_subdir / filename

            # Skip if file exists and no overwrite
            if output_path.exists() and not args.overwrite:
                print(f"Skipping {filename} - already exists")
                continue

            if file_type == "full":
                stats_json = generate_full_stats_json(metadata, organized_cases, args.max_attempts, args.max_rounds)
                # Write full stats file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_json, f, indent=2, ensure_ascii=False)
                print(f"Generated full stats file: {output_path}")
                print(f"  Total cases: {stats_json['summary']['total_cases']}")
                print(f"  Total attempts: {stats_json['summary']['total_attempts']}")
            else:  # minimal
                # Write config.json (only if not exists or overwrite)
                config_path = output_subdir / "config.json"
                if not config_path.exists() or args.overwrite:
                    config_json = generate_config_json(metadata, organized_cases)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config_json, f, indent=2, ensure_ascii=False)
                    print(f"Generated config file: {config_path}")

                # Write case success stats file
                case_stats = generate_minimal_stats_json(metadata, organized_cases, args.max_attempts, args.max_rounds)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(case_stats, f, indent=2, ensure_ascii=False)

                success_count = sum(1 for case_type in case_stats.values() for case in case_type.values() if case["stat_final_success"])
                total_cases = sum(1 for case_type in case_stats.values() for _ in case_type)

                print(f"Generated minimal stats file: {output_path}")
                print(f"  Total cases: {total_cases}")
                print(f"  Successful cases: {success_count}")

        processed_count += 1

    print(f"\nExtraction completed!")
    print(f"Processed files: {processed_count}")
    print(f"Skipped files: {skipped_count}")
    return 0


if __name__ == "__main__":
    exit(main())