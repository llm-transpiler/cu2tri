#!/usr/bin/env python3
"""
Fused Attention Benchmark Test
测试 triton_tutorials/fused_attention 实现
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

def test_fused_attention():
    print("🧪 Testing Fused Attention Benchmark")
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

        print(f"📋 Testing {len(all_test_data)} attention configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            batch_size, n_heads, seq_len, head_dim = shape
            cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_bundle

            print(f"\n📐 Configuration {i+1}/{total}: {shape}")

            try:
                # Run PyTorch reference
                torch_result = torch_kernel(*torch_all_inputs)

                # Run Triton kernel
                triton_result = triton_kernel(*cuda_all_inputs)

                # Compare results
                diff = (triton_result - torch_result).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                print(f"   Max difference: {max_diff:.6f}")
                print(f"   Mean difference: {mean_diff:.6f}")

                # Use appropriate tolerance for attention (FP16 + softmax)
                tolerance = 1e-1  # Higher tolerance due to softmax and FP16
                if max_diff < tolerance:
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED (tolerance: {tolerance})")

                # Simple performance check
                if torch.cuda.is_available():
                    # Warmup
                    for _ in range(2):  # Fewer warmups due to memory
                        _ = torch_kernel(*torch_all_inputs)
                        _ = triton_kernel(*cuda_all_inputs)
                    torch.cuda.synchronize()

                    # Benchmark PyTorch
                    torch_time = []
                    for _ in range(5):  # Fewer iterations for memory
                        start_time = time.perf_counter()
                        _ = torch_kernel(*torch_all_inputs)
                        torch.cuda.synchronize()
                        torch_time.append(time.perf_counter() - start_time)

                    # Benchmark Triton
                    triton_time = []
                    for _ in range(5):
                        start_time = time.perf_counter()
                        _ = triton_kernel(*cuda_all_inputs)
                        torch.cuda.synchronize()
                        triton_time.append(time.perf_counter() - start_time)

                    avg_torch_time = sum(torch_time) / len(torch_time) * 1000
                    avg_triton_time = sum(triton_time) / len(triton_time) * 1000
                    speedup = avg_torch_time / avg_triton_time if avg_triton_time > 0 else 0

                    # Calculate effective bandwidth/similar metric
                    total_elements = batch_size * n_heads * seq_len * head_dim
                    # Approximate FLOPs for attention: QK^T + Softmax + Attn*V
                    flops = 2 * seq_len * seq_len * head_dim + 2 * seq_len * seq_len + 2 * seq_len * head_dim
                    total_flops = batch_size * n_heads * flops
                    tflops = total_flops / (avg_triton_time / 1000) / 1e12 if avg_triton_time > 0 else 0

                    print(f"   📊 PyTorch: {avg_torch_time:.2f}ms")
                    print(f"   📊 Triton: {avg_triton_time:.2f}ms")
                    print(f"   🚀 Speedup: {speedup:.2f}x")
                    print(f"   📈 TFLOPS: {tflops:.2f}")

                # Clear memory to avoid OOM
                del torch_result, triton_result
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

        print(f"\n📊 Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All fused attention tests passed!")
            return True
        else:
            print("💥 Some tests failed!")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_fused_attention()
    sys.exit(0 if success else 1)