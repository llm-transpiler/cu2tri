#!/usr/bin/env python3
"""
RoPE Benchmark Test
测试 Liger-Kernel RoPE 实现
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

def test_rope():
    print("🧪 Testing Liger-Kernel RoPE Benchmark")
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

        print(f"📋 Testing {len(all_test_data)} RoPE configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

            # Handle both 4-element and 5-element shapes (for GQA)
            if len(shape) == 4:
                batch_size, seq_len, num_heads, head_dim = shape
                shape_str = f"({batch_size}, {seq_len}, {num_heads}, {head_dim})"
            else:
                batch_size, seq_len, num_heads, head_dim, num_kv_heads = shape
                shape_str = f"({batch_size}, {seq_len}, {num_heads}, {head_dim}, kv_heads={num_kv_heads})"

            print(f"\n📐 Configuration {i+1}/{total}: {shape_str}")

            try:
                # Unpack inputs
                q_cuda, k_cuda = cuda_all_inputs
                q_torch, k_torch = torch_all_inputs

                # Run PyTorch reference
                start_time = time.perf_counter()
                torch_q_out, torch_k_out = torch_kernel(q_torch, k_torch)
                torch_time = time.perf_counter() - start_time

                # Run Triton kernel
                start_time = time.perf_counter()
                triton_q_out, triton_k_out = triton_kernel(q_cuda, k_cuda)
                triton_time = time.perf_counter() - start_time

                # Compare results for both outputs
                # Compare query output
                diff_q = torch.abs(triton_q_out - torch_q_out)
                max_diff_q = diff_q.max().item()
                mean_diff_q = diff_q.mean().item()

                # Compare key output
                diff_k = torch.abs(triton_k_out - torch_k_out)
                max_diff_k = diff_k.max().item()
                mean_diff_k = diff_k.mean().item()

                # Relative differences
                rel_diff_q = diff_q / (torch.abs(torch_q_out) + 1e-8)
                max_rel_diff_q = rel_diff_q.max().item()

                rel_diff_k = diff_k / (torch.abs(torch_k_out) + 1e-8)
                max_rel_diff_k = rel_diff_k.max().item()

                print(f"   Query - Max abs diff: {max_diff_q:.8f}, Max rel diff: {max_rel_diff_q:.8f}")
                print(f"   Key   - Max abs diff: {max_diff_k:.8f}, Max rel diff: {max_rel_diff_k:.8f}")

                # Use appropriate tolerance for RoPE (due to floating point precision)
                abs_tolerance = 1e-4
                rel_tolerance = 1e-3

                if (max_diff_q < abs_tolerance or max_rel_diff_q < rel_tolerance) and \
                   (max_diff_k < abs_tolerance or max_rel_diff_k < rel_tolerance):
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
                n_elements = q_cuda.numel() + k_cuda.numel()
                bytes_per_element = 4  # float32
                bandwidth = (n_elements * bytes_per_element) / (triton_time / 1000) / 1e9
                print(f"   📈 Bandwidth:    {bandwidth:.2f} GB/s")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n📊 Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All RoPE tests passed!")
            return True
        else:
            print("💥 Some tests failed!")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_rope()
    sys.exit(0 if success else 1)