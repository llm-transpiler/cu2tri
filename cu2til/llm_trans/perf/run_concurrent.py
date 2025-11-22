import json
import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict

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


def load_case_success_stats(case_success_path: Path, case_type_filter: str = None, case_name_filter: str = None, test_all_attempts: bool = True) -> Dict:
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
                if test_all_attempts:
                    # Test ALL successful attempts
                    successful_attempts = [attempt for attempt in case_info.get("attempts", []) if attempt.get("success", False)]
                    for attempt in successful_attempts:
                        successful_cases.append({
                            "case_type": case_type,
                            "case_name": case_name,
                            "successful_attempt": attempt["attempt_number"],
                            "successful_round": attempt["round_final"]
                        })
                else:
                    # Only test the LAST successful attempt
                    successful_attempts = [attempt for attempt in case_info.get("attempts", []) if attempt.get("success", False)]
                    if successful_attempts:
                        # Sort by attempt_number and take the last one
                        last_attempt = max(successful_attempts, key=lambda x: x["attempt_number"])
                        successful_cases.append({
                            "case_type": case_type,
                            "case_name": case_name,
                            "successful_attempt": last_attempt["attempt_number"],
                            "successful_round": last_attempt["round_final"]
                        })

    return successful_cases


def discover_all_case_success_files(stats_root: Path, cu2tri_name: str = "cu2tri", xpiler_name: str = "xpiler",
                                     model_filter: str = None, timestamp_filter: str = None) -> List[tuple]:
    """Discover all case_success.json files"""
    case_success_files = []

    stats_base = stats_root / cu2tri_name / xpiler_name

    if not stats_base.exists():
        print(f"❌ Stats directory not found: {stats_base}")
        return case_success_files

    # Find model directories
    model_dirs = [d for d in stats_base.iterdir() if d.is_dir()]

    if model_filter:
        model_dirs = [d for d in model_dirs if d.name == model_filter]

    for model_dir in sorted(model_dirs):
        print(f"Scanning model: {model_dir.name}")

        # Find timestamp directories
        timestamp_dirs = []
        if timestamp_filter:
            timestamp_path = model_dir / timestamp_filter
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


async def run_single_test(test_info, gpu_id=None, timeout=600):
    """运行单个性能测试"""
    # 构建命令
    cmd = [
        "python3", "run_performance_tests.py",
        "--model", test_info["model"],
        "--timestamp", test_info["timestamp"],
        "--case-name", test_info["case_name"],
        "--case-type", test_info["case_type"],
        "--attempt-number", str(test_info["successful_attempt"])
    ]
    if gpu_id:
        cmd.extend(["--gpu", str(gpu_id)])

    try:
        # 运行测试
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent
        )

        # 添加超时控制
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return {
                "test_info": test_info,
                "returncode": -1,
                "stdout": stdout.decode(),
                "stderr": f"Test timed out after {timeout} seconds"
            }

        return {
            "test_info": test_info,
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }
    except Exception as e:
        return {
            "test_info": test_info,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Exception during test execution: {str(e)}"
        }


async def run_concurrent_tests(cases, max_concurrent=20, gpu_id=None, timeout=600):
    """并发运行多个测试"""
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(cases)

    # 使用线程安全的计数器
    from asyncio import Lock
    completed_lock = Lock()

    async def bounded_test(case):
        async with semaphore:
            # 显示启动信息，使用当前已完成的数量
            async with completed_lock:
                nonlocal completed
                current_completed = completed
                print(f"🔄 Testing {case['case_name']} (attempt {case['successful_attempt']}) [{current_completed + 1}/{total}]...", flush=True)

            result = await run_single_test(case, gpu_id, timeout)

            # 更新完成的计数器
            async with completed_lock:
                completed += 1
                current_completed = completed

            if result["returncode"] == 0:
                print(f"✅ [{current_completed}/{total}] Success: {result['test_info']['case_name']} (attempt {result['test_info']['successful_attempt']})", flush=True)
            else:
                error_msg = result['stderr'][:200] if result['stderr'] else 'No error message'
                print(f"❌ [{current_completed}/{total}] Failed: {result['test_info']['case_name']} - {error_msg}", flush=True)

            return result

    # 创建所有任务
    print(f"🚀 Starting {total} tests with up to {max_concurrent} concurrent tasks...", flush=True)
    tasks = [bounded_test(case) for case in cases]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    success = 0
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Exception: {result}", flush=True)
            failed += 1
        elif result["returncode"] == 0:
            success += 1
        else:
            failed += 1

    print(f"\n🏁 Concurrent testing completed!", flush=True)
    print(f"Successfully tested: {success}", flush=True)
    print(f"Failed to test: {failed}", flush=True)
    print(f"Total kernels: {total}", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concurrent performance testing for successful triton kernels")
    parser.add_argument("--model", help="Model name (optional)")
    parser.add_argument("--timestamp", help="Timestamp (optional)")
    parser.add_argument("--case-type", help="Case type filter (e.g., 'add')")
    parser.add_argument("--case-name", help="Specific case name (optional)")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent tests (default: 20)")
    parser.add_argument("--gpu", type=int, help="GPU ID (optional)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per test in seconds (default: 600)")
    parser.add_argument("--test-all-attempts", action="store_true", default=True, help="Test all successful attempts (default: True)")
    parser.add_argument("--test-last-attempt-only", action="store_true", help="Only test the last successful attempt per case")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested without running tests")

    args = parser.parse_args()

    # Handle attempt selection logic
    test_all_attempts = args.test_all_attempts and not args.test_last_attempt_only

    # Check if nvgpu server is running
    if not NVGPU_AVAILABLE:
        print("❌ Error: NVGPU client is not available")
        sys.exit(1)

    try:
        client = NVGPUClient("http://localhost:8080")
        if not client.health_check():
            print("❌ Error: NVGPU server is not responding correctly")
            sys.exit(1)
        print("✅ NVGPU server is running")
    except Exception as e:
        print(f"❌ Error: NVGPU server is not running: {e}")
        print("Please start it first.")
        sys.exit(1)

    # Discover and load all successful cases
    stats_root = Path("/data/apps/project/cu2tri/cu2til/llm_trans/stats")
    case_success_files = discover_all_case_success_files(
        stats_root,
        model_filter=args.model,
        timestamp_filter=args.timestamp
    )

    if not case_success_files:
        print("❌ No case_success.json files found")
        sys.exit(0)

    print(f"\n📊 Found {len(case_success_files)} case_success.json files")

    # Extract all successful cases with filtering
    all_successful_cases = []
    for case_success_file, file_info in case_success_files:
        print(f"\n📖 Reading: {case_success_file}")
        successful_cases = load_case_success_stats(
            case_success_file,
            args.case_type,
            args.case_name,
            test_all_attempts
        )
        print(f"  ✅ Found {len(successful_cases)} successful cases (filtered)")

        # Add file info to each case
        for case in successful_cases:
            case.update(file_info)

        all_successful_cases.extend(successful_cases)

    print(f"\n🎯 Total successful kernels to test: {len(all_successful_cases)}")

    if len(all_successful_cases) == 0:
        print("⚠️ No successful kernels found matching the specified filters.")
        sys.exit(0)

    # 显示将要运行的测试
    print(f"\n🔍 Kernels to test:")
    for case in all_successful_cases[:10]:  # 只显示前10个
        print(f"  ✅ {case['model']}/{case['timestamp']}/{case['case_name']} (attempt {case['successful_attempt']})")
    if len(all_successful_cases) > 10:
        print(f"  ... and {len(all_successful_cases) - 10} more")

    # Handle dry-run mode
    if args.dry_run:
        mode_info = "all successful attempts" if test_all_attempts else "last successful attempt only"
        print(f"\n🔍 DRY RUN - Would test {len(all_successful_cases)} kernels ({mode_info}) with the following settings:")
        print(f"Maximum concurrent tasks: {args.max_concurrent}")
        print(f"GPU: {args.gpu if args.gpu else 'auto-assigned'}")
        print(f"Attempt selection: {mode_info}")
        print(f"\nTo actually run the tests, remove --dry-run argument.")
        sys.exit(0)

    # 运行并发测试
    print(f"\n🚀 Starting concurrent performance testing...")
    print(f"Maximum concurrent tasks: {args.max_concurrent}")
    print(f"GPU: {args.gpu if args.gpu else 'auto-assigned'}")
    print(f"Total cases: {len(all_successful_cases)}")

    asyncio.run(run_concurrent_tests(all_successful_cases, args.max_concurrent, args.gpu, args.timeout))