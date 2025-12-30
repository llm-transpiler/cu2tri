#!/usr/bin/env python3
"""
Script to delete all 'perf' folders under the specified directory.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path

def delete_perf_folders(base_dir, dry_run=False):
    """
    Delete all 'perf' folders under the given base directory.

    Args:
        base_dir (str): Base directory to search for perf folders
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return False

    if not base_path.is_dir():
        print(f"Error: Base path is not a directory: {base_dir}")
        return False

    # Find all perf folders
    perf_folders = list(base_path.rglob("perf"))
    perf_folders = [p for p in perf_folders if p.is_dir()]

    if not perf_folders:
        print("No 'perf' folders found.")
        return True

    print(f"Found {len(perf_folders)} 'perf' folders:")

    deleted_count = 0
    failed_count = 0

    for perf_folder in sorted(perf_folders):
        print(f"  {perf_folder}")

        if not dry_run:
            try:
                shutil.rmtree(perf_folder)
                print(f"    ✓ Deleted")
                deleted_count += 1
            except Exception as e:
                print(f"    ✗ Failed to delete: {e}")
                failed_count += 1

    if dry_run:
        print(f"\nDry run mode: No folders were actually deleted.")
        print(f"Would delete {len(perf_folders)} 'perf' folders.")
    else:
        print(f"\nDeletion complete:")
        print(f"  Successfully deleted: {deleted_count}")
        print(f"  Failed to delete: {failed_count}")

    return failed_count == 0

def main():
    parser = argparse.ArgumentParser(description="Delete all 'perf' folders under a specified directory")
    parser.add_argument("base_dir",
                       help="Base directory to search for perf folders")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--confirm", "-y", action="store_true",
                       help="Skip confirmation prompt")

    args = parser.parse_args()

    # Default directory if not specified
    if not args.base_dir:
        args.base_dir = "/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/deepseek_v3_2_exp/20251024_075645 copy"

    # Show what will be deleted
    base_path = Path(args.base_dir)
    perf_folders = list(base_path.rglob("perf"))
    perf_folders = [p for p in perf_folders if p.is_dir()]

    print(f"Base directory: {args.base_dir}")
    print(f"Found {len(perf_folders)} 'perf' folders to delete")

    if not args.dry_run and not args.confirm:
        response = input("\nAre you sure you want to delete these folders? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return 1

    success = delete_perf_folders(args.base_dir, args.dry_run)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())