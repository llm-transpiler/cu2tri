import torch
import os
import sys
import argparse
from cu2til.tools.builder import load_cuda_kernel
from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform, cuda_input_tensor_to_ptr
from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results, run_performance_test

TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

def check_cuda_vs_torch(testname="MAXPOOL", enable_perf=False, compile_only=False):
    params = Params()
    print(f"📊 Test parameters: {testname} input=({params.batch_size}, {params.channels}, {params.height}, {params.width}), kernel={params.kernel_size}, stride={params.stride}")

    try:
        from get_data import get_cuda_argtypes
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=False)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    if compile_only:
        print(f"🔧 Compile-only mode: CUDA kernel compilation completed successfully")
        return

    print(f"📋 Creating GPU test data...")
    cuda_all_inputs, torch_all_inputs, cuda_output_tensors = get_cuda_torch_inputs(params)
    cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)

    output_torch = torch_kernel(*torch_all_inputs)

    print(f"⚡ Running CUDA kernel...")
    cuda_kernel(*cuda_all_inputs_ptr)

    output_cuda = cuda_output_tensor_transform(cuda_output_tensors[0])
    checkok = compare_results(output_torch, output_cuda)

    if enable_perf:
        run_performance_test(cuda_all_inputs_ptr, torch_all_inputs, cuda_kernel, torch_kernel)

    print(f"\n🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   CUDA    : {output_cuda.flatten()[:4]}")

def parse_args():
    parser = argparse.ArgumentParser(description='CUDA Add kernel test')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the CUDA kernel without running tests (default: False)')
    parser.add_argument('--no-perf', action='store_true',
                        help='Disable performance testing (default: False, performance testing enabled)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf
    compile_only = args.compile_only
    check_cuda_vs_torch(testname="Add", enable_perf=enable_perf, compile_only=compile_only)