import torch
import os
import sys
import argparse
from cu2til.tools.builder import load_cuda_kernel
from get_data import get_cuda_triton_inputs, get_cuda_torch_inputs, get_cuda_triton_torch_inputs, Params
from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results, run_performance_all, run_performance_test
from triton_.kernel import triton_kernel

def check_triton_all(test_root_dir=None, enable_perf=False, enable_cuda=False, enable_pytorch=False):
    if test_root_dir is None:
        raise ValueError("test_root_dir is required")
    sys.path.insert(0, test_root_dir)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Parameter settings (inferred from filename) 
    params = Params()

    # Determine test mode and prepare data accordingly
    if enable_cuda and enable_pytorch:
        return _test_cuda_pytorch_mode(test_root_dir, params, enable_perf)
    elif enable_cuda:
        return _test_cuda_only_mode(test_root_dir, params, enable_perf)
    elif enable_pytorch:
        return _test_pytorch_only_mode(test_root_dir, params, enable_perf)
    else:
        raise ValueError("At least one of CUDA or PyTorch must be enabled.")

def _test_cuda_only_mode(test_root_dir, params, enable_perf):
    """仅CUDA模式：Triton vs CUDA，先比正确性，再比性能"""
    print(f"📋 Running Triton vs CUDA comparison...")
    
    # Prepare data for CUDA format
    cuda_all_inputs, triton_all_inputs, cuda_output_tensors, triton_output_tensors = get_cuda_triton_inputs(params)
    
    # Load CUDA kernel
    try:
        from get_data import get_cuda_argtypes, cuda_input_tensor_to_ptr
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(test_root_dir, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return False
    
    # Run kernels for correctness testing
    print(f"⚡ Running CUDA kernel...")
    cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)
    cuda_kernel(*cuda_all_inputs_ptr)
    output_cuda = cuda_output_tensors[0]
    
    print(f"⚡ Running Triton kernel...")
    triton_kernel(*triton_all_inputs)
    output_triton = triton_output_tensors[0]
    
    # Correctness comparison
    print(f"🔍 Testing correctness: Triton vs CUDA")
    checkok = compare_results(output_cuda, output_triton, atol=1e-2, rtol=1e-2)
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   CUDA   : {output_cuda.flatten()[:4]}")
    print(f"   Triton : {output_triton.flatten()[:4]}")
    
    # Performance comparison if enabled
    if enable_perf:
        print(f"🚀 Running performance comparison...")
        run_performance_test(triton_all_inputs, cuda_all_inputs_ptr, triton_kernel, cuda_kernel, test_type=["Triton", "CUDA"])
    
    return checkok

def _test_pytorch_only_mode(test_root_dir, params, enable_perf):
    """仅PyTorch模式：Triton vs PyTorch，先比正确性，再比性能"""
    print(f"📋 Running Triton vs PyTorch comparison...")
    
    # Prepare data for PyTorch format (and derive Triton format)
    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    
    # Run kernels for correctness testing
    print(f"⚡ Running PyTorch kernel...")
    output_pytorch = torch_kernel(*torch_all_inputs)
    
    print(f"⚡ Running Triton kernel...")
    triton_kernel(*triton_all_inputs)
    output_triton = triton_output_tensors[0]
    
    # Correctness comparison
    print(f"🔍 Testing correctness: Triton vs PyTorch")
    checkok = compare_results(output_pytorch, output_triton, atol=1e-2, rtol=1e-2)
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   PyTorch: {output_pytorch.flatten()[:4]}")
    print(f"   Triton : {output_triton.flatten()[:4]}")
    
    # Performance comparison if enabled
    if enable_perf:
        print(f"🚀 Running performance comparison...")
        run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])
    
    return checkok

def _test_cuda_pytorch_mode(test_root_dir, params, enable_perf):
    """CUDA+PyTorch模式：Triton vs CUDA比正确性，三方性能对比"""
    print(f"📋 Running comprehensive comparison: Triton vs CUDA vs PyTorch...")
    
    cuda_all_inputs, triton_all_inputs, torch_all_inputs, cuda_output_tensors, triton_output_tensors = get_cuda_triton_torch_inputs(params)
    
    # Load CUDA kernel
    try:
        from get_data import get_cuda_argtypes, cuda_input_tensor_to_ptr
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(test_root_dir, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return False
    
    # Run all kernels
    print(f"⚡ Running PyTorch kernel...")
    output_pytorch = torch_kernel(*torch_all_inputs)
    
    print(f"⚡ Running CUDA kernel...")
    cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)
    cuda_kernel(*cuda_all_inputs_ptr)
    output_cuda = cuda_output_tensors[0]
    
    print(f"⚡ Running Triton kernel...")
    triton_kernel(*triton_all_inputs)
    output_triton = triton_output_tensors[0]
    
    # Correctness comparison: Triton vs CUDA
    print(f"🔍 Testing correctness: Triton vs CUDA")
    checkok = compare_results(output_cuda, output_triton, atol=1e-2, rtol=1e-2)
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   CUDA   : {output_cuda.flatten()[:4]}")
    print(f"   PyTorch: {output_pytorch.flatten()[:4]}")
    print(f"   Triton : {output_triton.flatten()[:4]}")
    
    # Performance comparison if enabled
    if enable_perf:
        print(f"🚀 Running comprehensive performance comparison...")
        run_performance_all(triton_all_inputs, cuda_all_inputs_ptr, torch_all_inputs, triton_kernel, cuda_kernel, torch_kernel)
    
    return checkok

def parse_args():
    TESTCASE_ROOT_DIR = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='Triton Conv2D kernel test')
    parser.add_argument('--no-perf', action='store_true',
                       help='Disable performance testing (default: False, performance testing enabled)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA testing (default: False, CUDA testing enabled)')
    parser.add_argument('--no-pytorch', action='store_true',
                       help='Disable PyTorch testing (default: False, PyTorch testing enabled)')
    parser.add_argument('--test-root-dir', type=str, default=TESTCASE_ROOT_DIR,
                       help='Test root directory (default: current directory)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    enable_cuda = not args.no_cuda  # Default is True, disable with --no-cuda
    enable_pytorch = not args.no_pytorch  # Default is True, disable with --no-pytorch
    test_root_dir = args.test_root_dir
    # enable_cuda = False
    # enable_pytorch = False
    if not enable_cuda and not enable_pytorch:
        raise ValueError("At least one of CUDA or PyTorch must be enabled.")
    check_triton_all(test_root_dir=test_root_dir, enable_perf=enable_perf, enable_cuda=enable_cuda, enable_pytorch=enable_pytorch)
