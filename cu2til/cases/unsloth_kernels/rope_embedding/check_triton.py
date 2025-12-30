#!/usr/bin/env python3
"""
Test script for unsloth rope_embedding kernel
"""
import sys
import os
import torch

# Add current directory to path
TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

from get_data import test_cases
from torch_.ref import torch_rope_embedding
from triton_.kernel import triton_rope_embedding

def test_unsloth_rope():
    """Test unsloth rope_embedding kernel correctness"""
    test_case_list = test_cases()
    print(f"Testing {len(test_case_list)} cases...")

    all_passed = True
    for i, (Q, cos, sin) in enumerate(test_case_list):
        try:
            # Triton result
            triton_result = triton_rope_embedding(Q, cos, sin)

            # PyTorch reference
            torch_result = torch_rope_embedding(Q, cos, sin)

            # Check correctness
            if not torch.allclose(triton_result, torch_result, rtol=1e-3, atol=1e-3):
                print(f"❌ Test case {i} failed!")
                print(f"  Input shapes: Q={Q.shape}, cos={cos.shape}, sin={sin.shape}")
                print(f"  Max diff: {(triton_result - torch_result).abs().max().item()}")
                all_passed = False
            else:
                print(f"✅ Test case {i} passed: Q{Q.shape} -> {triton_result.shape}")

        except Exception as e:
            print(f"❌ Test case {i} failed with exception: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_unsloth_rope()
    if success:
        print("\n🎉 All unsloth rope_embedding tests passed!")
    else:
        print("\n💥 Some unsloth rope_embedding tests failed!")
    sys.exit(0 if success else 1)