#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark Test
测试 triton_tutorials/matmul 实现
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

def test_matmul():
    print("🧪 Testing Matrix Multiplication Benchmark")
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

        print(f"📋 Testing {len(all_test_data)} matrix configurations")

        passed = 0
        total = len(all_test_data)

        for i, (shape, test_bundle) in enumerate(all_test_data):
            M, N, K = shape
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

                # Use appropriate tolerance for FP16
                tolerance = 1e-2  # FP16 needs higher tolerance
                if max_diff < tolerance:
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print(f"   ❌ FAILED (tolerance: {tolerance})")

                # Simple performance check
                if torch.cuda.is_available():
                    # Warmup
                    for _ in range(5):
                        _ = torch_kernel(*torch_all_inputs)
                        _ = triton_kernel(*cuda_all_inputs)
                    torch.cuda.synchronize()

                    # Benchmark PyTorch
                    torch_time = []
                    for _ in range(20):
                        start_time = time.perf_counter()
                        _ = torch_kernel(*torch_all_inputs)
                        torch.cuda.synchronize()
                        torch_time.append(time.perf_counter() - start_time)

                    # Benchmark Triton
                    triton_time = []
                    for _ in range(20):
                        start_time = time.perf_counter()
                        _ = triton_kernel(*cuda_all_inputs)
                        torch.cuda.synchronize()
                        triton_time.append(time.perf_counter() - start_time)

                    avg_torch_time = sum(torch_time) / len(torch_time) * 1000
                    avg_triton_time = sum(triton_time) / len(triton_time) * 1000
                    speedup = avg_torch_time / avg_triton_time

                    flops = 2 * M * N * K
                    torch_tflops = flops / (avg_torch_time / 1000) / 1e12
                    triton_tflops = flops / (avg_triton_time / 1000) / 1e12

                    print(f"   📊 PyTorch: {avg_torch_time:.2f}ms ({torch_tflops:.2f} TFLOPS)")
                    print(f"   📊 Triton: {avg_triton_time:.2f}ms ({triton_tflops:.2f} TFLOPS)")
                    print(f"   🚀 Speedup: {speedup:.2f}x")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n📊 Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All matrix multiplication tests passed!")
            return True
        else:
            print("💥 Some tests failed!")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_matmul()
    sys.exit(0 if success else 1)