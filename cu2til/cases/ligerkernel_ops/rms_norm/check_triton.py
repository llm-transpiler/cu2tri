#!/usr/bin/env python3
"""
Test script for ligerkernel rms_norm kernel
"""
import sys
import os
import torch

# Add current directory to path
TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

from get_data import test_cases
from torch_.ref import torch_rms_norm
from triton_.kernel import triton_rms_norm

def test_ligerkernel_rms_norm():
    """Test ligerkernel rms_norm kernel correctness"""
    test_case_list = test_cases()
    print(f"Testing {len(test_case_list)} cases...")

    all_passed = True
    for i, (hidden_states, weight) in enumerate(test_case_list):
        try:
            # Triton result
            triton_result = triton_rms_norm(hidden_states, weight)

            # PyTorch reference
            torch_result = torch_rms_norm(hidden_states, weight)

            # Check correctness
            if not torch.allclose(triton_result, torch_result, rtol=1e-2, atol=1e-2):
                print(f"❌ Test case {i} failed!")
                print(f"  Input shapes: hidden_states={hidden_states.shape}, weight={weight.shape}")
                print(f"  Max diff: {(triton_result - torch_result).abs().max().item()}")
                all_passed = False
            else:
                print(f"✅ Test case {i} passed: {hidden_states.shape} -> {triton_result.shape}")

        except Exception as e:
            print(f"❌ Test case {i} failed with exception: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_ligerkernel_rms_norm()
    if success:
        print("\n🎉 All ligerkernel rms_norm tests passed!")
    else:
        print("\n💥 Some ligerkernel rms_norm tests failed!")
    sys.exit(0 if success else 1)