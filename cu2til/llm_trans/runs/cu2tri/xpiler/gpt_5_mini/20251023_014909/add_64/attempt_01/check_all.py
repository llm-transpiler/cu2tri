import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, get_cuda_argtypes  # type: ignore
try:
    from get_data import cuda_output_tensor_transform  # type: ignore
except ImportError:
    cuda_output_tensor_transform = lambda x: x

from torch_.ref import torch_kernel  # type: ignore
from triton_.kernel import triton_kernel  # type: ignore
from cu2til.tools.checker import check_all_kernels

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive kernel test: CUDA, Triton vs PyTorch')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the CUDA kernel without running tests (default: False)')
    parser.add_argument('--no-perf', action='store_true',
                        help='Disable performance testing (default: False, performance testing enabled)')
    parser.add_argument('--correctness', action='store_true',
                        help='Enable correctness testing (default: False when neither flag set)')
    parser.add_argument('--performance', action='store_true', 
                        help='Enable performance testing (default: False when neither flag set)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Determine test modes based on memory requirements
    if not args.correctness and not args.performance:
        # When neither flag is set, only build (compile-only mode)
        compile_only = True
        enable_perf = False
    else:
        # Use explicit flags when set
        compile_only = args.compile_only
        enable_perf = args.performance and not args.no_perf
    
    params = Params()
    
    success = check_all_kernels(
        testcase_root_dir=TESTCASE_ROOT_DIR,
        get_cuda_torch_inputs=get_cuda_torch_inputs,
        params=params,
        torch_kernel=torch_kernel,
        triton_kernel=triton_kernel,
        get_cuda_argtypes=get_cuda_argtypes,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=enable_perf,
        compile_only=compile_only
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
