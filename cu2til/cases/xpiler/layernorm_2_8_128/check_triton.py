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

def check_triton_vs_torch(testname="LayerNorm", enable_perf=False):
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Parameter settings (inferred from filename) 
    params = Params()

    print(f"📋 Creating GPU test data...")
    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    output_torch = torch_kernel(*torch_all_inputs)
    
    # Run Triton implementation
    print(f"⚡ Running Triton kernel...")
    
    # Call Triton kernel
    triton_kernel(*triton_all_inputs)

    # transform the output tensor to the specific layout
    output_triton = cuda_output_tensor_transform(triton_output_tensors[0])
    checkok = compare_results(output_torch, output_triton, atol=1e-4, rtol=1e-3)  # relaxed for numerical precision
    if enable_perf:
        run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   Triton  : {output_triton.flatten()[:4]}")
    
    return checkok

def parse_args():
    parser = argparse.ArgumentParser(description='Triton LayerNorm kernel test')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing (default: False, performance testing enabled)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    check_triton_vs_torch(testname="LayerNorm", enable_perf=enable_perf)
