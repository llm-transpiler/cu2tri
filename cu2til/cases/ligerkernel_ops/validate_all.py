#!/usr/bin/env python3
"""
Comprehensive Liger-Kernel Benchmarks Validation Script
验证所有已创建的liger-kernel benchmarks
"""
import sys
import os
import time
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = None
for candidate in Path(__file__).resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_benchmark_test(benchmark_path: str) -> dict:
    """运行单个benchmark测试"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing {benchmark_path}")
    print(f"{'='*80}")

    test_file = None
    for possible_name in ["test_*.py"]:
        test_files = list(Path(benchmark_path).glob(possible_name))
        if test_files:
            test_file = test_files[0]
            break

    if not test_file:
        return {"status": "SKIP", "reason": "No test file found"}

    try:
        # Run test script
        cmd = [
            "bash", "-c",
            f"source /data/apps/miniforge3/etc/profile.d/conda.sh && conda activate serve && python {test_file}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse results
            output = result.stdout
            if "All" in output and "tests passed!" in output:
                return {"status": "PASS", "output": output}
            elif "tests passed" in output or "PASSED" in output:
                # Extract passed/total counts
                lines = output.split('\n')
                result_line = [l for l in lines if "Results:" in l and "tests passed" in l]
                if result_line:
                    return {"status": "PARTIAL", "output": output}
                else:
                    return {"status": "PASS", "output": output}
            else:
                return {"status": "FAIL", "output": output, "error": result.stderr}
        else:
            return {"status": "FAIL", "output": result.stdout, "error": result.stderr}

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "reason": "Test timed out after 300s"}
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}

def validate_all_liger_kernels():
    """验证所有liger-kernel benchmarks"""
    print("🚀 Comprehensive Liger-Kernel Benchmarks Validation")
    print("This script will test all available liger-kernel benchmarks")

    # Benchmark directories to test
    benchmarks_root = PROJECT_ROOT / "cu2til/cases/ligerkernel_ops"
    benchmark_dirs = []

    for item in benchmarks_root.iterdir():
        if item.is_dir() and (item / "get_data.py").exists():
            benchmark_dirs.append(item.name)

    benchmark_dirs.sort()

    print(f"\n📋 Found {len(benchmark_dirs)} benchmarks: {benchmark_dirs}")

    results = {}
    passed = 0
    total = len(benchmark_dirs)

    for benchmark_name in benchmark_dirs:
        benchmark_path = benchmarks_root / benchmark_name
        result = run_benchmark_test(str(benchmark_path))
        results[benchmark_name] = result

        if result["status"] == "PASS":
            passed += 1
            print(f"✅ {benchmark_name}: PASSED")
        elif result["status"] == "PARTIAL":
            passed += 0.5
            print(f"⚠️  {benchmark_name}: PARTIAL")
        elif result["status"] == "SKIP":
            print(f"⏭️  {benchmark_name}: SKIPPED")
        else:
            print(f"❌ {benchmark_name}: FAILED")

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 LIGER-KERNEL BENCHMARKS VALIDATION SUMMARY")
    print(f"{'='*80}")

    for benchmark_name, result in results.items():
        status = result["status"]
        if status == "PASS":
            print(f"✅ {benchmark_name:20} : PASSED")
        elif status == "PARTIAL":
            print(f"⚠️  {benchmark_name:20} : PARTIAL")
        elif status == "SKIP":
            print(f"⏭️  {benchmark_name:20} : SKIPPED - {result.get('reason', 'Unknown')}")
        elif status == "TIMEOUT":
            print(f"⏱️  {benchmark_name:20} : TIMEOUT")
        else:
            print(f"❌ {benchmark_name:20} : FAILED - {result.get('reason', 'Unknown')}")

    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\n🎯 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")

    if passed == total:
        print("🎉 All liger-kernel benchmarks passed!")
        return True
    elif passed >= total * 0.8:
        print("👍 Most liger-kernel benchmarks passed!")
        return True
    else:
        print("💥 Many liger-kernel benchmarks failed!")
        return False

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate liger-kernel benchmarks')
    parser.add_argument('--comprehensive', action='store_true', default=True, help='Run comprehensive validation')

    args = parser.parse_args()

    if args.comprehensive:
        success = validate_all_liger_kernels()
        return 0 if success else 1
    else:
        validate_all_liger_kernels()
        return 0

if __name__ == "__main__":
    sys.exit(main())