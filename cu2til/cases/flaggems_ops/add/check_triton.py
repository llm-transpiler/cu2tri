#!/usr/bin/env python3
"""
Test script for flaggems add kernel
"""
import sys
import os
import torch

# Add current directory to path
TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

from get_data import get_test_cases
from torch_.ref import torch_kernel
from triton_.kernel import binary_add_tensor

def test_flaggems_add():
    """Test flaggems add kernel correctness"""
    test_cases = get_test_cases()
    print(f"Testing {len(test_cases)} cases...")

    all_passed = True
    for i, (x, y) in enumerate(test_cases):
        try:
            # Triton result
            triton_result = binary_add_tensor(x, y)

            # PyTorch reference
            torch_result = torch_kernel(x, y)

            # Check correctness
            if not torch.allclose(triton_result, torch_result, rtol=1e-3, atol=1e-3):
                print(f"❌ Test case {i} failed!")
                print(f"  Input shapes: {x.shape}, {y.shape}")
                print(f"  Max diff: {(triton_result - torch_result).abs().max().item()}")
                all_passed = False
            else:
                print(f"✅ Test case {i} passed: {x.shape} + {y.shape} = {triton_result.shape}")

        except Exception as e:
            print(f"❌ Test case {i} failed with exception: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_flaggems_add()
    if success:
        print("\n🎉 All flaggems add tests passed!")
    else:
        print("\n💥 Some flaggems add tests failed!")
    sys.exit(0 if success else 1)