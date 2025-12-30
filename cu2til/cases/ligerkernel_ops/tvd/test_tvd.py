#!/usr/bin/env python3
"""
Total Variation Distance Benchmark Test
测试 Liger-Kernel TVD 实现
"""
import sys
import os
import time
import torch
from pathlib import Path

PROJECT_ROOT = None
for candidate in Path(__file__).resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_tvd():
    print("🧪 Testing Liger-Kernel Total Variation Distance Benchmark")
    print("=" * 60)

    case_dir = Path(__file__).parent
    sys.path.insert(0, str(case_dir))

    try:
        from get_data import Params, get_all_cuda_torch_inputs
        from torch_.ref import torch_kernel
        from triton_.kernel import triton_kernel

        params = Params()
        all_test_data = get_all_cuda_torch_inputs(params)

        print(f"📋 Testing {len(all_test_data)} TVD configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            cuda_all_inputs, torch_all_inputs, _ = test_bundle

            print(f"\n📐 Configuration {i+1}/{total}: {shape}")

            try:
                p_cuda, q_cuda = cuda_all_inputs
                p_torch, q_torch = torch_all_inputs

                start_time = time.perf_counter()
                torch_result = torch_kernel(p_torch, q_torch, reduction="batchmean")
                torch_time = time.perf_counter() - start_time

                start_time = time.perf_counter()
                triton_result = triton_kernel(p_cuda, q_cuda, reduction="batchmean")
                triton_time = time.perf_counter() - start_time

                diff = torch.abs(triton_result - torch_result)
                max_diff = diff.item() if diff.numel() == 1 else diff.max().item()
                rel_diff = diff / (torch.abs(torch_result) + 1e-8)
                max_rel_diff = rel_diff.item() if rel_diff.numel() == 1 else rel_diff.max().item()

                print(f"   Abs diff: {max_diff:.8f}, Rel diff: {max_rel_diff:.8f}")

                abs_tol, rel_tol = 1e-5, 1e-4
                if max_diff < abs_tol or max_rel_diff < rel_tol:
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED")

                print(f"   📊 PyTorch: {torch_time*1000:.2f}ms, Triton: {triton_time*1000:.2f}ms")
                if triton_time > 0:
                    print(f"   🚀 Speedup: {torch_time/triton_time:.2f}x")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")

        print(f"\n📊 Results: {passed}/{total} tests passed")
        return passed == total

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_tvd()
    sys.exit(0 if success else 1)