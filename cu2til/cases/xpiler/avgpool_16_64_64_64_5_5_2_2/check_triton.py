import torch
import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform
from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results, run_performance_test
from triton_.kernel import triton_kernel

TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

def check_triton_vs_torch(testname="AvgPool2D", enable_perf=False):
    params = Params()
    print(f"📊 Test parameters: {testname} input=({params.batch_size}, {params.channels}, {params.height}, {params.width}), kernel={params.kernel_size}, stride={params.stride}")

    print(f"📋 Creating GPU test data...")
    cuda_all_inputs, torch_all_inputs, cuda_output_tensors = get_cuda_torch_inputs(params)
    
    output_torch = torch_kernel(*torch_all_inputs)

    print(f"⚡ Running Triton kernel...")
    # Triton kernel directly takes torch tensors
    output_triton = triton_kernel(torch_all_inputs[0], torch_all_inputs[1])

    checkok = compare_results(output_torch, output_triton)

    if enable_perf:
        run_performance_test(None, torch_all_inputs, None, torch_kernel, triton_kernel=triton_kernel, test_type=["Triton", "PyTorch"])

    print(f"\n🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   Triton  : {output_triton.flatten()[:4]}")

def parse_args():
    parser = argparse.ArgumentParser(description='Triton Add kernel test')
    parser.add_argument('--no-perf', action='store_true',
                        help='Disable performance testing (default: False, performance testing enabled)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf
    check_triton_vs_torch(testname="Add", enable_perf=enable_perf)
