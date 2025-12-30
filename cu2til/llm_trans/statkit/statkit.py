#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main():
    """Main entry point for statkit - provides easy access to common operations"""

    statkit_dir = Path(__file__).parent

    print("🧮 CU2TRI Statistics Toolkit")
    print("=" * 50)
    print()
    print("Available operations:")
    print("1. Extract stats from single file/directory")
    print("2. Batch process all timestamps")
    print("3. Show help for extract_stats.py")
    print("4. Show help for batch_process.py")
    print()

    try:
        choice = input("Select operation (1-4): ").strip()

        if choice == "1":
            # Single file extraction
            path = input("Enter JSONL file or directory path (press Enter for default runs/cu2tri): ").strip()
            max_attempts = input("Max attempts (optional): ").strip()
            max_rounds = input("Max rounds (optional): ").strip()
            overwrite = input("Overwrite existing files? (y/N): ").strip().lower() == 'y'
            minimal_only = input("Generate minimal stats only? (y/N): ").strip().lower() == 'y'

            cmd = ["python3", str(statkit_dir / "extract_stats.py")]
            if path:
                cmd.append(path)
            if max_attempts:
                cmd.extend(["--max-attempts", max_attempts])
            if max_rounds:
                cmd.extend(["--max-rounds", max_rounds])
            if overwrite:
                cmd.append("--overwrite")
            if minimal_only:
                cmd.append("--minimal-only")

            subprocess.run(cmd)

        elif choice == "2":
            # Batch processing
            model = input("Filter by model (optional): ").strip()
            timestamp = input("Filter by timestamp (optional): ").strip()
            max_attempts = input("Max attempts (optional): ").strip()
            max_rounds = input("Max rounds (optional): ").strip()
            overwrite = input("Overwrite existing files? (y/N): ").strip().lower() == 'y'
            dry_run = input("Dry run (show what would be processed)? (y/N): ").strip().lower() == 'y'
            minimal_only = input("Generate minimal stats only? (y/N): ").strip().lower() == 'y'

            cmd = ["python3", str(statkit_dir / "batch_process.py")]
            if model:
                cmd.extend(["--model", model])
            if timestamp:
                cmd.extend(["--timestamp", timestamp])
            if max_attempts:
                cmd.extend(["--max-attempts", max_attempts])
            if max_rounds:
                cmd.extend(["--max-rounds", max_rounds])
            if overwrite:
                cmd.append("--overwrite")
            if dry_run:
                cmd.append("--dry-run")
            if minimal_only:
                cmd.append("--minimal-only")

            subprocess.run(cmd)

        elif choice == "3":
            subprocess.run(["python3", str(statkit_dir / "extract_stats.py"), "--help"])

        elif choice == "4":
            subprocess.run(["python3", str(statkit_dir / "batch_process.py"), "--help"])

        else:
            print("Invalid choice")
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())