import os
import sys
import argparse
from get_data import get_all_cuda_torch_inputs, Params, cuda_output_tensor_transform  # type: ignore
from torch_.ref import torch_kernel  # type: ignore
from triton_.kernel import triton_kernel  # type: ignore
from cu2til.tools.checker import check_triton_vs_torch_dynamic

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Triton kernel test for multiple shapes')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing (default: False, performance testing enabled)')
    parser.add_argument('--shapes', nargs='+', type=int, 
                        help='Custom shapes to test (e.g., --shapes 512 1024 2048). If not provided, uses default test shapes.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    
    params = Params()
    
    success = check_triton_vs_torch_dynamic(
        get_all_cuda_torch_inputs=get_all_cuda_torch_inputs,
        params=params,
        torch_kernel=torch_kernel,
        triton_kernel=triton_kernel,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=enable_perf
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
