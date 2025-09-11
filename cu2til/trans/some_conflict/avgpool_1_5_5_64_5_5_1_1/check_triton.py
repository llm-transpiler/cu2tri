import torch
import os
import sys
import argparse
from get_data import get_cuda_torch_inputs, Params, cuda_output_tensor_transform, cuda_input_tensor_to_ptr
from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results, run_performance_test
from triton_.kernel import triton_kernel

TESTCASE_ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTCASE_ROOT_DIR)

def check_triton_vs_torch(testname="AvgPool2D", enable_perf=False):
    params = Params()
    print(f"📊 Test parameters: {testname} input=({params.batch_size}, {params.channels}, {params.height}, {params.width}), kernel={params.kernel_size}, stride={params.stride}")

    print(f"📋 Creating GPU test data...")
    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    
    output_torch = torch_kernel(*torch_all_inputs)

    print(f"⚡ Running Triton kernel...")
    # Triton kernel directly takes torch tensors
    triton_kernel(*triton_all_inputs)

    # output_triton = cuda_output_tensor_transform(triton_output_tensors[0])
    output_triton = triton_output_tensors[0]
    checkok = compare_results(output_torch, output_triton)

    if enable_perf and checkok:
        run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])

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
