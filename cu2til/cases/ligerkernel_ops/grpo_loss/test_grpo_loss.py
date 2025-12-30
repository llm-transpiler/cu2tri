#!/usr/bin/env python3
"""GRPO Loss Test"""
import sys
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

def test_grpo_loss():
    print("🧪 Testing GRPO Loss Benchmark")
    print("=" * 60)

    case_dir = Path(__file__).parent
    sys.path.insert(0, str(case_dir))

    try:
        from get_data import Params, get_all_cuda_torch_inputs
        from torch_.ref import torch_kernel
        from triton_.kernel import triton_kernel

        params = Params()
        all_test_data = get_all_cuda_torch_inputs(params)
        print(f"📋 Testing {len(all_test_data)} configurations")

        passed = 0
        for i, (shape, (cuda_inputs, torch_inputs, _)) in enumerate(all_test_data):
            print(f"\n📐 Config {i+1}/{len(all_test_data)}: {shape}")
            try:
                logits_cuda, adv_cuda = cuda_inputs
                logits_torch, adv_torch = torch_inputs

                torch_result = torch_kernel(logits_torch, adv_torch)
                triton_result = triton_kernel(logits_cuda, adv_cuda)

                diff = torch.abs(triton_result - torch_result)
                max_diff = diff.item()
                rel_diff = diff / (torch.abs(torch_result) + 1e-8)
                max_rel_diff = rel_diff.item()

                print(f"   Abs: {max_diff:.8f}, Rel: {max_rel_diff:.8f}")

                abs_tol, rel_tol = 1e-3, 1e-2
                if max_diff < abs_tol or max_rel_diff < rel_tol:
                    passed += 1
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")

                print(f"   📊 PyTorch: {time.perf_counter()*1000:.2f}ms")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")

        print(f"\n📊 Results: {passed}/{len(all_test_data)} passed")
        return passed == len(all_test_data)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_grpo_loss()
    sys.exit(0 if success else 1)