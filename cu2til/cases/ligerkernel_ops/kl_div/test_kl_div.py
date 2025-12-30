#!/usr/bin/env python3
"""
KL Divergence Loss Benchmark Test
测试 Liger-Kernel KL Divergence Loss 实现
"""
import sys
import os
import time
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = None
for candidate in Path(__file__).resolve().parents:
    if (candidate / "cu2til").exists():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_kl_div():
    print("🧪 Testing Liger-Kernel KL Divergence Loss Benchmark")
    print("=" * 60)

    # Add current case directory to path
    case_dir = Path(__file__).parent
    sys.path.insert(0, str(case_dir))

    try:
        # Import test data
        from get_data import Params, get_all_cuda_torch_inputs
        from torch_.ref import torch_kernel
        from triton_.kernel import triton_kernel

        params = Params()
        all_test_data = get_all_cuda_torch_inputs(params)

        print(f"📋 Testing {len(all_test_data)} KL Divergence configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

            print(f"\n📐 Configuration {i+1}/{total}: {shape}")

            try:
                # Unpack inputs
                pred_cuda, target_cuda = cuda_all_inputs
                pred_torch, target_torch = torch_all_inputs

                # Run PyTorch reference
                start_time = time.perf_counter()
                torch_result = torch_kernel(pred_torch, target_torch, reduction="batchmean")
                torch_time = time.perf_counter() - start_time

                # Run Triton kernel
                start_time = time.perf_counter()
                triton_result = triton_kernel(pred_cuda, target_cuda, reduction="batchmean")
                triton_time = time.perf_counter() - start_time

                # Compare results
                diff = torch.abs(triton_result - torch_result)
                max_diff = diff.item() if diff.numel() == 1 else diff.max().item()

                # Relative difference
                rel_diff = diff / (torch.abs(torch_result) + 1e-8)
                max_rel_diff = rel_diff.item() if rel_diff.numel() == 1 else rel_diff.max().item()

                print(f"   Absolute diff: {max_diff:.8f}")
                print(f"   Relative diff: {max_rel_diff:.8f}")

                # Use appropriate tolerance for KL divergence
                abs_tolerance = 1e-3
                rel_tolerance = 1e-2

                if max_diff < abs_tolerance or max_rel_diff < rel_tolerance:
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED (abs_tol: {abs_tolerance}, rel_tol: {rel_tolerance})")
                    print(f"   PyTorch result: {torch_result.item():.8f}")
                    print(f"   Triton result:  {triton_result.item():.8f}")

                # Performance comparison
                print(f"   📊 PyTorch time: {torch_time*1000:.2f}ms")
                print(f"   📊 Triton time:  {triton_time*1000:.2f}ms")
                if triton_time > 0:
                    speedup = torch_time / triton_time
                    print(f"   🚀 Speedup:     {speedup:.2f}x")

                # Compute memory bandwidth
                n_elements = pred_cuda.numel() + target_cuda.numel()
                bytes_per_element = 4  # float32
                bandwidth = (n_elements * bytes_per_element) / (triton_time / 1000) / 1e9
                print(f"   📈 Bandwidth:    {bandwidth:.2f} GB/s")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n📊 Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All KL Divergence tests passed!")
            return True
        else:
            print("💥 Some tests failed!")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_kl_div()
    sys.exit(0 if success else 1)