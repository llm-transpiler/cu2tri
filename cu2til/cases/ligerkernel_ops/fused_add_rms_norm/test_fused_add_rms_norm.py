#!/usr/bin/env python3
"""
Fused Add RMS Norm Benchmark Test
测试 Liger-Kernel Fused Add RMS Norm 实现
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

def test_fused_add_rms_norm():
    print("🧪 Testing Liger-Kernel Fused Add RMS Norm Benchmark")
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

        print(f"📋 Testing {len(all_test_data)} Fused Add RMS Norm configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

            print(f"\n📐 Configuration {i+1}/{total}: {shape}")

            try:
                # Unpack inputs
                hidden_cuda, residual_cuda, weight_cuda = cuda_all_inputs
                hidden_torch, residual_torch, weight_torch = torch_all_inputs

                # Run PyTorch reference
                start_time = time.perf_counter()
                torch_normalized, torch_residual = torch_kernel(hidden_torch, residual_torch, weight_torch)
                torch_time = time.perf_counter() - start_time

                # Run Triton kernel
                start_time = time.perf_counter()
                triton_normalized, triton_residual = triton_kernel(hidden_cuda, residual_cuda, weight_cuda)
                triton_time = time.perf_counter() - start_time

                # Compare results for both outputs
                # Compare normalized output
                diff_norm = torch.abs(triton_normalized - torch_normalized)
                max_diff_norm = diff_norm.max().item()
                mean_diff_norm = diff_norm.mean().item()

                # Compare residual output
                diff_res = torch.abs(triton_residual - torch_residual)
                max_diff_res = diff_res.max().item()
                mean_diff_res = diff_res.mean().item()

                # Relative differences
                rel_diff_norm = diff_norm / (torch.abs(torch_normalized) + 1e-8)
                max_rel_diff_norm = rel_diff_norm.max().item()

                rel_diff_res = diff_res / (torch.abs(torch_residual) + 1e-8)
                max_rel_diff_res = rel_diff_res.max().item()

                print(f"   Normalized - Max abs diff: {max_diff_norm:.8f}, Max rel diff: {max_rel_diff_norm:.8f}")
                print(f"   Residual   - Max abs diff: {max_diff_res:.8f}, Max rel diff: {max_rel_diff_res:.8f}")

                # Use appropriate tolerance for RMS norm
                abs_tolerance = 1e-5
                rel_tolerance = 1e-4

                if (max_diff_norm < abs_tolerance or max_rel_diff_norm < rel_tolerance) and \
                   (max_diff_res < abs_tolerance or max_rel_diff_res < rel_tolerance):
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED (abs_tol: {abs_tolerance}, rel_tol: {rel_tolerance})")

                # Performance comparison
                print(f"   📊 PyTorch time: {torch_time*1000:.2f}ms")
                print(f"   📊 Triton time:  {triton_time*1000:.2f}ms")
                if triton_time > 0:
                    speedup = torch_time / triton_time
                    print(f"   🚀 Speedup:     {speedup:.2f}x")

                # Compute memory bandwidth
                n_elements = hidden_cuda.numel() * 2 + weight_cuda.numel()  # hidden + residual + weight
                bytes_per_element = 4  # float32
                bandwidth = (n_elements * bytes_per_element) / (triton_time / 1000) / 1e9
                print(f"   📈 Bandwidth:    {bandwidth:.2f} GB/s")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n📊 Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All Fused Add RMS Norm tests passed!")
            return True
        else:
            print("💥 Some tests failed!")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_fused_add_rms_norm()
    sys.exit(0 if success else 1)