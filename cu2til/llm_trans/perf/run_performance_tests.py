#!/usr/bin/env python3

import json
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import re
import sys

# Add parent directories to path to import llm_trans modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Add the project root to path
    project_root = Path(__file__).parent.parent.parent
    server_nvgpu_path = project_root / "server" / "nvgpu"
    if server_nvgpu_path.exists():
        sys.path.insert(0, str(server_nvgpu_path))

    from server.nvgpu.client import NVGPUClient
    NVGPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NVGPU client not available: {e}")
    NVGPU_AVAILABLE = False


def load_case_success_stats(case_success_path: Path, case_type_filter: str = None, case_name_filter: str = None) -> Dict:
    """Load case_success.json and extract successful attempts with optional filtering"""
    with open(case_success_path, 'r') as f:
        data = json.load(f)

    successful_cases = []

    # Iterate through case types and cases
    for case_type, cases in data.items():
        # Apply case type filter if specified
        if case_type_filter:
            if case_type != case_type_filter:
                continue

        for case_name, case_info in cases.items():
            # Apply case name filter if specified
            if case_name_filter:
                if case_name != case_name_filter:
                    continue

            if case_info.get("stat_final_success", False):
                # Find ALL successful attempts - test each one
                successful_attempts = [attempt for attempt in case_info.get("attempts", []) if attempt.get("success", False)]
                for attempt in successful_attempts:
                    successful_cases.append({
                        "case_type": case_type,
                        "case_name": case_name,
                        "successful_attempt": attempt["attempt_number"],
                        "successful_round": attempt["round_final"]
                    })

    return successful_cases


def find_kernel_path(runs_root: Path, model_name: str, timestamp: str, case_name: str, attempt_number: int) -> Optional[Path]:
    """Find the triton kernel path for a specific case and attempt"""
    # Navigate to the case directory structure: runs/cu2tri/xpiler/{model}/{timestamp}/{case_name}/attempt_{num}/triton_/kernel.py
    case_dir = runs_root / "cu2tri" / "xpiler" / model_name / timestamp / case_name

    # Find the specific attempt directory
    attempt_dir_name = f"attempt_{attempt_number:02d}"
    attempt_dir = case_dir / attempt_dir_name

    kernel_path = attempt_dir / "triton_" / "kernel.py"

    if kernel_path.exists():
        return kernel_path

    return None


def discover_all_case_success_files(stats_root: Path, cu2tri_name: str = "cu2tri", xpiler_name: str = "xpiler",
                                   model_name: str = None, timestamp: str = None) -> List[Tuple[Path, Dict]]:
    """Discover all case_success.json files and filter by model/timestamp if specified"""

    case_success_files = []

    # Navigate to stats/cu2tri/xpiler
    cu2tri_root = stats_root / cu2tri_name / xpiler_name

    if not cu2tri_root.exists():
        print(f"Error: {cu2tri_root} does not exist")
        return case_success_files

    # Get all model directories
    model_dirs = []
    if model_name:
        model_path = cu2tri_root / model_name
        if model_path.exists():
            model_dirs.append(model_path)
    else:
        model_dirs = [d for d in cu2tri_root.iterdir() if d.is_dir()]

    for model_dir in sorted(model_dirs):
        print(f"Scanning model: {model_dir.name}")

        # Get all timestamp directories
        timestamp_dirs = []
        if timestamp:
            timestamp_path = model_dir / timestamp
            if timestamp_path.exists():
                timestamp_dirs.append(timestamp_path)
        else:
            timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

        for timestamp_dir in sorted(timestamp_dirs):
            case_success_file = timestamp_dir / "case_success.json"
            if case_success_file.exists():
                case_success_files.append((case_success_file, {
                    "model": model_dir.name,
                    "timestamp": timestamp_dir.name
                }))
                print(f"  Found: {model_dir.name}/{timestamp_dir.name}")

    return case_success_files


def submit_performance_test(kernel_path: Path, test_info: Dict, gpu_id: Optional[int] = None, nvgpu_server: str = "http://localhost:8080") -> Optional[str]:
    """Submit kernel to nvgpu server for performance testing using correct NVGPU client"""

    if not NVGPU_AVAILABLE:
        print("  ❌ NVGPU client not available")
        return None

    try:
        # Initialize NVGPU client
        client = NVGPUClient(nvgpu_server)

        # Check health
        if not client.health_check():
            print(f"  ❌ NVGPU server at {nvgpu_server} is not responding")
            return None

        # Submit performance test task using the correct method
        task_label = f"perf_test/{test_info['model']}/{test_info['case_name']}/attempt_{test_info.get('successful_attempt', 1)}"

        # Find the performance test script in the attempt directory - prioritize GPU performance scripts
        perf_script_path = kernel_path.parent.parent / "check_triton_gpu_all.py"
        if not perf_script_path.exists():
            # Try alternative GPU script
            perf_script_path = kernel_path.parent.parent / "check_triton.py"
        if not perf_script_path.exists():
            # Try general check script
            perf_script_path = kernel_path.parent.parent / "check_all.py"
        if not perf_script_path.exists():
            # Try the torch_ directory
            perf_script_path = kernel_path.parent.parent / "torch_" / "check_triton_vs_torch.py"
        if not perf_script_path.exists():
            # Default to the attempt directory
            perf_script_path = kernel_path.parent.parent

        print(f"  📝 Using performance script: {perf_script_path}")

        task_id = client.submit_task_in_script_dir(
            script_path=str(perf_script_path.absolute()),
            task_mode="shared",       # Use shared mode for concurrent performance testing
            task_type="performance",  # Still a performance task
            task_label=task_label,
            gpu_id=gpu_id,
        )

        return task_id

    except Exception as e:
        print(f"  ❌ Failed to submit to NVGPU server: {e}")
        return None


def wait_for_performance_completion(task_id: str, nvgpu_server: str = "http://localhost:8080", timeout: int = 1200) -> Dict:
    """Wait for performance test completion with longer timeout using NVGPU client"""

    if not NVGPU_AVAILABLE:
        return {
            "status": "error",
            "error": "NVGPU client not available",
            "task_id": task_id
        }

    try:
        client = NVGPUClient(nvgpu_server)
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = client.get_task(task_id)

            if result.status in ["completed", "failed", "error", "cancelled"]:
                return result

            time.sleep(10)  # Poll every 10 seconds for performance tests

        return {
            "status": "timeout",
            "error": f"Performance test timed out after {timeout} seconds",
            "task_id": task_id
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Error waiting for task completion: {e}",
            "task_id": task_id
        }


def get_task_logs(task_id: str, nvgpu_server: str = "http://localhost:8080") -> Dict[str, str]:
    """Get stdout and stderr logs from completed task"""

    if not NVGPU_AVAILABLE:
        return {"stdout": "NVGPU client not available", "stderr": "NVGPU client not available"}

    try:
        client = NVGPUClient(nvgpu_server)

        stdout_log = client.get_full_task_log(task_id, log_type="stdout")
        stderr_log = client.get_full_task_log(task_id, log_type="stderr")

        return {
            "stdout": stdout_log,
            "stderr": stderr_log
        }

    except Exception as e:
        return {
            "stdout": f"Failed to get stdout log: {e}",
            "stderr": f"Failed to get stderr log: {e}"
        }


def save_performance_logs(results: List[Dict], gpu_id: Optional[int] = None):
    """Save performance test logs to logs/perf directories with proper format"""

    for result in results:
        if result.get("status") == "completed" and result.get("task_id"):
            task_id = result["task_id"]
            kernel_path = Path(result["kernel_path"])

            # Get logs and task result
            logs = get_task_logs(task_id)
            task_result = get_task_result(task_id)

            # Save to attempt logs/perf directory
            logs_dir = kernel_path.parent.parent / "logs" / "perf"
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Save logs with proper format similar to triton_test_round logs
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            perf_log_path = logs_dir / f"performance_test_{timestamp}.log"

            with open(perf_log_path, 'w') as f:
                f.write(f"=== Performance Test (NVGPU) ===\n")
                f.write(f"TASK: perf_test/{result['model']}/{result['case_name']}/attempt_{result.get('attempt_number', 1)}::[{task_id}]\n")
                f.write(f"Task ID: {task_id}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Model: {result['model']}\n")
                f.write(f"Case: {result['case_name']}\n")
                f.write(f"Attempt: {result.get('attempt_number', 1)}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Exit code: {getattr(task_result, 'exit_code', 'N/A')}\n")
                f.write(f"\n=== Timing Information ===\n")

                # Extract timing information from task result
                if hasattr(task_result, 'total_duration_ms'):
                    f.write(f"Total elapsed time (submit to finish) : {task_result.total_duration_ms:.3f} ms\n")
                if hasattr(task_result, 'running_duration_ms'):
                    f.write(f"Execution time (actual run)           : {task_result.running_duration_ms:.3f} ms\n")
                if hasattr(task_result, 'pending_duration_ms'):
                    f.write(f"Pending time (submit to queue)        : {task_result.pending_duration_ms:.3f} ms\n")
                if hasattr(task_result, 'queue_duration_ms'):
                    f.write(f"Queue time (assign to start)          : {task_result.queue_duration_ms:.3f} ms\n")
                if hasattr(task_result, 'waiting_duration_ms'):
                    f.write(f"Waiting time (queue)                  : {task_result.waiting_duration_ms:.3f} ms\n")

                # Add timestamps if available
                if hasattr(task_result, 'submit_time'):
                    f.write(f"Submit time : {task_result.submit_time}\n")
                if hasattr(task_result, 'start_time'):
                    f.write(f"Start time  : {task_result.start_time}\n")
                if hasattr(task_result, 'end_time'):
                    f.write(f"End time    : {task_result.end_time}\n")

                f.write(f"\nServer log: {getattr(task_result, 'log_file', 'N/A')}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(logs["stdout"])
                f.write("\n=== STDERR ===\n")
                f.write(logs["stderr"])
                f.write("\n=== END OF LOG ===\n")

            print(f"  📄 Performance log saved to: {perf_log_path}")


def get_task_result(task_id: str, nvgpu_server: str = "http://localhost:8080"):
    """Get complete task result with timing information"""

    if not NVGPU_AVAILABLE:
        return None

    try:
        client = NVGPUClient(nvgpu_server)
        return client.get_task(task_id)
    except Exception as e:
        print(f"  ⚠️ Could not get task result for {task_id}: {e}")
        return None


def save_performance_results(results: List[Dict], output_dir: Path, gpu_id: Optional[int] = None):
    """Save performance test results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate results filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gpu_suffix = f"_gpu{gpu_id}" if gpu_id is not None else "_auto"
    results_file = output_dir / f"performance_results{gpu_suffix}_{timestamp}.json"

    # Sort results by model, timestamp, case_type, case_name
    results.sort(key=lambda x: (x["model"], x["timestamp"], x["case_type"], x["case_name"]))

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Performance results saved to: {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Performance testing for successful triton kernels")
    parser.add_argument("--stats-root", default="/data/apps/project/cu2tri/cu2til/llm_trans/stats",
                       help="Root directory for stats")
    parser.add_argument("--runs-root", default="/data/apps/project/cu2tri/cu2til/llm_trans/runs",
                       help="Root directory for runs")
    parser.add_argument("--cu2tri", default="cu2tri", help="CU2TRI directory name")
    parser.add_argument("--xpiler", default="xpiler", help="Xpiler directory name")
    parser.add_argument("--model", help="Process only specific model")
    parser.add_argument("--timestamp", help="Process only specific timestamp")
    parser.add_argument("--case-type", help="Process only specific case type (e.g., 'add' for all add cases)")
    parser.add_argument("--case-name", help="Process only specific case name (e.g., 'add_1_15_64' for specific case)")
    parser.add_argument("--gpu", type=int, help="GPU ID for shared mode (optional - let NVGPU server auto-assign if not specified)")
    parser.add_argument("--output-dir", default="/data/apps/project/cu2tri/cu2til/llm_trans/perf/results",
                       help="Output directory for performance results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested without running tests")

    args = parser.parse_args()

    # Validate case filtering arguments
    if args.case_type and args.case_name:
        # If both are specified, ensure case_name starts with case_type
        if not args.case_name.startswith(args.case_type):
            print(f"Error: case_name '{args.case_name}' doesn't start with case_type '{args.case_type}'")
            return 1

    gpu_info = f"GPU ID: {args.gpu} (shared mode)" if args.gpu else "GPU: auto-assigned (shared mode)"
    print(f"Performance Testing for Successful Triton Kernels")
    print(f"===============================================")
    print(f"{gpu_info}")
    print(f"Stats root: {args.stats_root}")
    print(f"Runs root: {args.runs_root}")

    if args.case_type:
        print(f"Case type filter: {args.case_type}")
    if args.case_name:
        print(f"Case name filter: {args.case_name}")

    # Check if nvgpu server is running
    if not NVGPU_AVAILABLE:
        print("❌ Error: NVGPU client is not available")
        return 1

    try:
        client = NVGPUClient("http://localhost:8080")
        if not client.health_check():
            print("❌ Error: NVGPU server is not responding correctly")
            return 1
        print("✅ NVGPU server is running")
    except Exception as e:
        print(f"❌ Error: NVGPU server is not running: {e}")
        print("Please start it first.")
        return 1

    # Discover all case_success.json files
    case_success_files = discover_all_case_success_files(
        Path(args.stats_root),
        args.cu2tri,
        args.xpiler,
        args.model,
        args.timestamp
    )

    if not case_success_files:
        print("❌ No case_success.json files found")
        return 0

    print(f"\n📊 Found {len(case_success_files)} case_success.json files")

    # Extract all successful cases with filtering
    all_successful_cases = []
    total_kernels = 0

    for case_success_file, file_info in case_success_files:
        print(f"\n📖 Reading: {case_success_file}")
        successful_cases = load_case_success_stats(
            case_success_file,
            args.case_type,
            args.case_name
        )
        print(f"  ✅ Found {len(successful_cases)} successful cases (filtered)")

        # Add file info to each case
        for case in successful_cases:
            case.update(file_info)
            total_kernels += 1

        all_successful_cases.extend(successful_cases)

    print(f"\n🎯 Total successful kernels to test: {total_kernels}")

    if total_kernels == 0:
        print("⚠️ No successful kernels found matching the specified filters.")
        return 0

    if args.dry_run:
        print("\n🔍 Dry run - kernels that would be tested:")
        for case in all_successful_cases:
            kernel_path = find_kernel_path(
                Path(args.runs_root),
                case["model"],
                case["timestamp"],
                case["case_name"],
                case["successful_attempt"]
            )
            if kernel_path:
                print(f"  ✅ {case['model']}/{case['timestamp']}/{case['case_name']} (attempt {case['successful_attempt']})")
            else:
                print(f"  ❌ {case['model']}/{case['timestamp']}/{case['case_name']} - kernel not found")
        print(f"\n🏁 Dry run completed. {total_kernels} kernels would be tested.")
        return 0

    # Run performance tests
    print(f"\n🚀 Starting performance testing...")
    print(f"Using GPU {args.gpu} in exclusive mode")

    results = []
    success_count = 0
    failure_count = 0

    for i, case in enumerate(all_successful_cases, 1):
        print(f"\n[{i}/{total_kernels}] Testing {case['case_name']} from {case['model']}/{case['timestamp']}...")

        # Find the kernel path
        kernel_path = find_kernel_path(
            Path(args.runs_root),
            case["model"],
            case["timestamp"],
            case["case_name"],
            case["successful_attempt"]
        )

        if not kernel_path:
            print(f"  ❌ Kernel not found for {case['case_name']} (attempt {case['successful_attempt']})")
            failure_count += 1
            results.append({
                "model": case["model"],
                "timestamp": case["timestamp"],
                "case_type": case["case_type"],
                "case_name": case["case_name"],
                "attempt_number": case["successful_attempt"],
                "successful_round": case["successful_round"],
                "kernel_path": None,
                "status": "kernel_not_found",
                "error": "Kernel file not found",
                "performance_metrics": None
            })
            continue

        print(f"  📂 Found kernel: {kernel_path}")

        # Submit performance test
        task_id = submit_performance_test(kernel_path, case, args.gpu)

        if not task_id:
            failure_count += 1
            results.append({
                "model": case["model"],
                "timestamp": case["timestamp"],
                "case_type": case["case_type"],
                "case_name": case["case_name"],
                "attempt_number": case["successful_attempt"],
                "successful_round": case["successful_round"],
                "kernel_path": str(kernel_path),
                "status": "submission_failed",
                "error": "Failed to submit to NVGPU server",
                "performance_metrics": None
            })
            continue

        print(f"  📤 Task submitted: {task_id}")

        # Wait for completion
        print(f"  ⏳ Waiting for performance test completion...")
        final_result = wait_for_performance_completion(task_id)

        # Record results
        result_data = {
            "model": case["model"],
            "timestamp": case["timestamp"],
            "case_type": case["case_type"],
            "case_name": case["case_name"],
            "attempt_number": case["successful_attempt"],
            "successful_round": case["successful_round"],
            "kernel_path": str(kernel_path),
            "task_id": task_id,
            "gpu_id": args.gpu,
            "status": final_result.status,
            "error": getattr(final_result, 'error', None),
            "performance_metrics": None
        }

        if final_result.status == "completed":
            print(f"  ✅ Performance test completed successfully")
            # Extract performance metrics if available
            perf_metrics = getattr(final_result, 'performance_metrics', {})
            if perf_metrics:
                result_data["performance_metrics"] = perf_metrics
                print(f"     Execution time: {perf_metrics.get('execution_time_ms', 'N/A')} ms")
                print(f"     Memory usage: {perf_metrics.get('memory_usage_mb', 'N/A')} MB")
            success_count += 1
        else:
            error_msg = getattr(final_result, 'error', 'Unknown error')
            print(f"  ❌ Performance test failed: {error_msg}")
            failure_count += 1

        results.append(result_data)

        # Save log for this specific test immediately (save to logs/perf)
        if final_result.status == "completed":
            print(f"  📄 Saving performance log for {case['case_name']} attempt {case['successful_attempt']}...")
            save_performance_logs([result_data], args.gpu)

        # Incrementally save results after each test (in case of interruption)
        save_performance_results(results, Path(args.output_dir), args.gpu)

    # Final save (already saved incrementally, but ensure everything is saved)
    save_performance_results(results, Path(args.output_dir), args.gpu)

    # Save any remaining logs that weren't saved during the loop
    print(f"\n📄 Saving any remaining performance logs...")
    save_performance_logs(results, args.gpu)

    print(f"\n🏁 Performance testing completed!")
    print(f"Successfully tested: {success_count}")
    print(f"Failed to test: {failure_count}")
    print(f"Total kernels: {total_kernels}")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    exit(main())