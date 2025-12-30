#!/usr/bin/env python3
"""
Independent test script for ligerkernel rms_norm kernel
"""
import sys
import os
import torch

# Force clear any cached imports
if 'get_data' in sys.modules:
    del sys.modules['get_data']
if 'torch_.ref' in sys.modules:
    del sys.modules['torch_.ref']
if 'triton_.kernel' in sys.modules:
    del sys.modules['triton_.kernel']

# Add current directory to path first
TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

# Clear Python cache
import importlib
import glob
for module_name in list(sys.modules.keys()):
    if 'ligerkernel' in module_name or 'rms_norm' in module_name:
        del sys.modules[module_name]

# Import fresh modules
from get_data import test_cases
from torch_.ref import torch_rms_norm
from triton_.kernel import triton_rms_norm

def test_ligerkernel_rms_norm():
    """Test ligerkernel rms_norm kernel correctness"""
    print("=== Liger-Kernel RMS Norm Test ===")
    test_case_list = test_cases()
    print(f"Testing {len(test_case_list)} cases...")

    all_passed = True
    for i, (hidden_states, weight) in enumerate(test_case_list):
        try:
            print(f"\nTest case {i}: hidden_states={hidden_states.shape}, weight={weight.shape}")

            # Triton result
            triton_result = triton_rms_norm(hidden_states, weight)

            # PyTorch reference
            torch_result = torch_rms_norm(hidden_states, weight)

            # Check correctness
            diff = (triton_result - torch_result).abs()
            max_diff = diff.max().item()

            # Use higher tolerance for numerical stability
            tolerance = 1e-2
            if max_diff > tolerance:
                print(f"❌ Test case {i} failed! Max diff: {max_diff:.6f}")
                all_passed = False
            else:
                print(f"✅ Test case {i} passed! Max diff: {max_diff:.6f}")

        except Exception as e:
            print(f"❌ Test case {i} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_ligerkernel_rms_norm()
    if success:
        print("\n🎉 All ligerkernel rms_norm tests passed!")
    else:
        print("\n💥 Some ligerkernel rms_norm tests failed!")
    sys.exit(0 if success else 1)