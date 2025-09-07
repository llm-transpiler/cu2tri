import torch
import numpy as np
import ctypes
import os
import time
import subprocess
import sys
from torch.nn import functional as F
from cu2til.tools.builder import compile_cuda_kernel, CUDA_FOLDER_NAME, SEED, load_cuda_kernel

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

import sys
sys.path.insert(0, TESTCASE_ROOT_DIR)

from torch_.ref import torch_kernel
from cu2til.tools.checker import compare_results

def get_inputs():
    """Create test data"""
    torch.manual_seed(SEED)
    batch_size, seq_len, num_heads, head_dim = 1, 4096, 12, 512
    
    # Create data directly on specified device
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device="cuda")
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device="cuda")
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32, device="cuda")
    return Q, K, V

def run_performance_test(Q, K, V, cuda_kernel):
    """Run GPU performance test"""
    print(f"\n🚀 GPU performance test:")
    
    from eval_.common.benchmark import benchmark_kernel
    torch_gpu_avg = benchmark_kernel(torch_kernel, (Q, K, V))
    
    """Call CUDA MHA kernel - directly use GPU tensor"""
    batch_size, seq_len, num_heads, head_dim = Q.shape
    
    # Ensure input tensors are on GPU and contiguous
    Q_gpu = Q.cuda().contiguous()
    K_gpu = K.cuda().contiguous()
    V_gpu = V.cuda().contiguous()
    
    # Create output tensor
    output_gpu = torch.empty_like(Q_gpu)
    
    # Get GPU pointers
    Q_ptr = Q_gpu.data_ptr()
    K_ptr = K_gpu.data_ptr()
    V_ptr = V_gpu.data_ptr()
    output_ptr = output_gpu.data_ptr()
    cuda_avg = benchmark_kernel(cuda_kernel, (Q_ptr, K_ptr, V_ptr, output_ptr, batch_size, seq_len, num_heads, head_dim), warmup=20, iterations=50)
    print(f"\n📊 GPU performance comparison:")
    print(f"  PyTorch (GPU): {torch_gpu_avg:7.3f} ms")
    print(f"  CUDA kernel:   {cuda_avg:7.3f} ms")
    
    if cuda_avg > 0:
        speedup = torch_gpu_avg / cuda_avg
        print(f"  CUDA speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
    return torch_gpu_avg, cuda_avg

def main():
    print("🚀 MHA CUDA automatic test")
    print("=" * 55)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Parameter settings (inferred from filename)
    batch_size, seq_len, num_heads, head_dim = 1, 4096, 12, 512
    print(f"📊 Test parameters: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    
    # Automatically compile and load CUDA library
    try:
        argtypes = [
            ctypes.c_void_p,  # d_queries (GPU pointer)
            ctypes.c_void_p,  # d_keys (GPU pointer)
            ctypes.c_void_p,  # d_values (GPU pointer)
            ctypes.c_void_p,  # d_output (GPU pointer)
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # num_heads
            ctypes.c_int      # head_dim
        ]
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes, force_compile=True)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return

    print(f"\n📋 Creating GPU test data...")
    inputs = get_inputs()
    Q = inputs[0]
    output_torch = torch_kernel(*inputs[:3])
    # Run CUDA implementation
    print(f"\n⚡ Running CUDA kernel...")
    batch_size, seq_len, num_heads, head_dim = Q.shape
    # Create output tensor
    output_cuda = torch.empty_like(Q)
    
    # Get GPU pointers
    inputs_ptr = [input.cuda().contiguous().data_ptr() for input in inputs]
    output_ptr = output_cuda.data_ptr()
    
    # Call CUDA kernel
    cuda_kernel(*inputs_ptr, output_ptr, batch_size, seq_len, num_heads, head_dim)
    compare_results(output_torch, output_cuda, atol=1e-4)
    run_performance_test(*inputs, cuda_kernel)
    
    # Display sample results
    print(f"\n🔬 Sample Output (first 5 values):")
    print(f"   PyTorch: {output_torch[0,0,0,:5]}")
    print(f"   CUDA:    {output_cuda[0,0,0,:5]}")
    
if __name__ == "__main__":
    main()
