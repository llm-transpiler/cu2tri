#!/usr/bin/env python3
"""
Quick validation script for all Liger-Kernel ops
快速验证所有已实现的Liger-Kernel Ops
"""
import sys
import os
from pathlib import Path

def validate_op(op_name):
    """Validate a single op"""
    print(f"\n🔍 Validating {op_name}...")
    op_dir = Path(__file__).parent / op_name
    test_file = op_dir / f"test_{op_name}.py"

    if not test_file.exists():
        print(f"❌ Missing test file: {test_file}")
        return False

    try:
        # Run a simple validation by checking imports
        sys.path.insert(0, str(op_dir))
        from get_data import Params, get_all_cuda_torch_inputs
        from torch_.ref import torch_kernel
        from triton_.kernel import triton_kernel

        params = Params()
        test_data = get_all_cuda_torch_inputs(params)
        print(f"✅ {op_name}: Structure valid, {len(test_data)} test cases")
        return True

    except Exception as e:
        print(f"❌ {op_name}: Validation failed - {e}")
        return False

def main():
    print("🚀 Quick Validation of All Liger-Kernel Ops")
    print("=" * 60)

    ops_dir = Path(__file__).parent
    all_ops = [d.name for d in ops_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

    passed = 0
    total = len(all_ops)

    print(f"📋 Found {total} ops to validate")

    for op in sorted(all_ops):
        if validate_op(op):
            passed += 1

    print(f"\n📊 Validation Summary: {passed}/{total} ops passed")
    if passed == total:
        print("🎉 All ops structurally valid!")
        return True
    else:
        print("💥 Some ops have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)