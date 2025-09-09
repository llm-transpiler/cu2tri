import torch
from eval_.common.benchmark import benchmark_kernel

SEED = 46
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
BOLD = "\033[1m"
END = "\033[0m"
YELLOW = ""
RED = ""
BLUE = ""
GREEN = ""
BOLD = ""
END = ""
def compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2):
    """Compare results from two implementations"""
    # Ensure both tensors are on the same device for comparison
    if output_torch.device != output_cuda.device:
        output_torch = output_torch.to(output_cuda.device)
    
    output_cuda = output_cuda.to(torch.float32)
    output_torch = output_torch.to(torch.float32)
    diff = torch.abs(output_torch - output_cuda)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    rel_err = diff / torch.maximum(torch.abs(output_torch), torch.tensor(1e-9, device=output_torch.device, dtype=torch.float32))
    max_rel_err = torch.max(rel_err)
    mean_rel_err = torch.mean(rel_err)
    
    # Print header
    print("\n" + "="*80)
    print("🔍 TENSOR COMPARISON RESULTS")
    print("="*80)
    
    # Print tensor shape info
    shape_info = f"Tensor Shape: {tuple(output_torch.shape)}, Total Elements: {output_torch.numel():,}"
    print(f"📊 {shape_info}")
    print("-" * 80)
    
    # Print error statistics in a nice table format
    print("📈 ERROR STATISTICS:")
    print("-" * 45)
    print(f"{'Metric':<20} {'Value':<15} {'Scientific':<15}")
    print("-" * 45)
    print(f"{'Max_Abs_Err':<20} {max_diff:<15.8f} {max_diff:<15.2e}")
    print(f"{'Mean_Abs_Err':<20} {mean_diff:<15.8f} {mean_diff:<15.2e}")
    print(f"{'Max_Rel_Err':<20} {max_rel_err:<15.8f} {max_rel_err:<15.2e}")
    print(f"{'Mean_Rel_Err':<20} {mean_rel_err:<15.8f} {mean_rel_err:<15.2e}")
    print("-" * 45)
    
    # Find top 4 absolute errors
    diff_flat = diff.flatten()
    abs_err_values, abs_err_indices = torch.topk(diff_flat, min(4, diff_flat.numel()))
    print("\n🔥 TOP 4 ABSOLUTE ERRORS:")
    print("-" * 80)
    
    # Print headers
    headers = [f"#{i+1}" for i in range(len(abs_err_values))]
    print(f"{'Metric':<15} {' '.join([f'{h:>18}' for h in headers])}")
    print("-" * 80)
    
    # Print error values
    errors = [f"{val:.2e}" for val in abs_err_values]
    abs_err_line = f"{'Abs_Err':<15} {' '.join([f'{e:>18}' for e in errors])}"
    print(f"{RED}{abs_err_line}{END}")
    # Print relative errors for absolute error top 4
    rel_errors_for_abs = [f"{rel_err.flatten()[idx]:.2e}" for idx in abs_err_indices]
    print(f"{'Rel_Err':<15} {' '.join([f'{e:>18}' for e in rel_errors_for_abs])}")
    
    
    # Print PyTorch values
    torch_vals = [f"{output_torch.flatten()[idx]:.8f}" for idx in abs_err_indices]
    print(f"{'PyTorch':<15} {' '.join([f'{v:>18}' for v in torch_vals])}")
    
    # Print CUDA values
    cuda_vals = [f"{output_cuda.flatten()[idx]:.8f}" for idx in abs_err_indices]
    print(f"{'CUDA':<15} {' '.join([f'{v:>18}' for v in cuda_vals])}")
    
    # Print differences
    diffs = [f"{output_torch.flatten()[idx] - output_cuda.flatten()[idx]:.8f}" for idx in abs_err_indices]
    print(f"{'Diff':<15} {' '.join([f'{d:>18}' for d in diffs])}")
    
    # Print indices
    indices = [f"{idx}" for idx in abs_err_indices]
    print(f"{'Index':<15} {' '.join([f'{idx:>18}' for idx in indices])}")
    
    # Find top 4 relative errors  
    rel_err_flat = rel_err.flatten()
    rel_err_values, rel_err_indices = torch.topk(rel_err_flat, min(4, rel_err_flat.numel()))
    print("\n📊 TOP 4 RELATIVE ERRORS:")
    print("-" * 80)
    
    # Print headers
    headers = [f"#{i+1}" for i in range(len(rel_err_values))]
    print(f"{'Metric':<15} {' '.join([f'{h:>18}' for h in headers])}")
    print("-" * 80)
    
    # Print absolute differences
    abs_diffs = [f"{torch.abs(output_torch.flatten()[idx] - output_cuda.flatten()[idx]):.2e}" for idx in rel_err_indices]
    print(f"{'Abs_Err':<15} {' '.join([f'{d:>18}' for d in abs_diffs])}")
    
    # Print relative error values
    rel_errors = [f"{val:.2e}" for val in rel_err_values]
    rel_err_line = f"{'Rel_Err':<15} {' '.join([f'{e:>18}' for e in rel_errors])}"
    print(f"{RED}{rel_err_line}{END}")
    
    # Print PyTorch values
    torch_vals = [f"{output_torch.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{'PyTorch':<15} {' '.join([f'{v:>18}' for v in torch_vals])}")
    
    # Print CUDA values
    cuda_vals = [f"{output_cuda.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{'CUDA':<15} {' '.join([f'{v:>18}' for v in cuda_vals])}")
    
    # Print differences
    diffs = [f"{output_torch.flatten()[idx] - output_cuda.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{'Diff':<15} {' '.join([f'{d:>18}' for d in diffs])}")
    
    # Print indices
    indices = [f"{idx}" for idx in rel_err_indices]
    print(f"{'Index':<15} {' '.join([f'{idx:>18}' for idx in indices])}")
    
    # Final result
    print("\n" + "="*80)
    is_close = torch.allclose(output_torch, output_cuda, atol=atol, rtol=rtol)
    if is_close:
        print(f"✅ RESULT: Tensors match within tolerance (atol={atol:.0e}, rtol={rtol:.0e})")
        status = "PASSED"
    else:
        print(f"❌ RESULT: Tensors do NOT match within tolerance (atol={atol:.0e}, rtol={rtol:.0e})")
        status = "FAILED"
    
    print(f"🎯 STATUS: {status}")
    print("="*80 + "\n")
    
    return is_close


def run_performance_test(test_all_inputs_ptr, baseline_all_inputs, test_kernel, baseline_kernel, test_type=["CUDA", "PyTorch"]):
    """Run GPU performance test"""
    print(f"🚀 GPU performance test:")
    baseline_ms = benchmark_kernel(baseline_kernel, baseline_all_inputs)
    test_kernel_ms = benchmark_kernel(test_kernel, test_all_inputs_ptr)

    print(f"📊 GPU performance comparison:")
    print(f"  {test_type[1]:<10}: {baseline_ms:8.3f} ms")
    print(f"  {test_type[0]:<10}: {test_kernel_ms:8.3f} ms")

    if test_kernel_ms > 0:
        speedup = baseline_ms / test_kernel_ms
        print(f"  {test_type[0]:<10} vs {test_type[1]:<10} speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")

    return baseline_ms, test_kernel_ms

def run_performance_all(triton_all_inputs, cuda_all_inputs, pytorch_all_inputs, triton_kernel, cuda_kernel, pytorch_kernel):
    """Run GPU performance test"""
    print(f"🚀 Performance Test:")
    triton_kernel_ms = benchmark_kernel(triton_kernel, triton_all_inputs)
    cuda_kernel_ms = benchmark_kernel(cuda_kernel, cuda_all_inputs)
    pytorch_kernel_ms = benchmark_kernel(pytorch_kernel, pytorch_all_inputs)

    print(f"📊 Performance Comparison:")
    print(f"  CUDA    : {cuda_kernel_ms:8.3f} ms")
    print(f"  Triton  : {triton_kernel_ms:8.3f} ms")
    print(f"  PyTorch : {pytorch_kernel_ms:8.3f} ms")

    if triton_kernel_ms > 0:
        speedup = cuda_kernel_ms / triton_kernel_ms
        print(f"  Triton vs CUDA    speedup:    {speedup:.3f}x {'🚀' if speedup > 1 else '📉'}")
        speedup = pytorch_kernel_ms / triton_kernel_ms
        print(f"  Triton vs PyTorch speedup:    {speedup:.3f}x {'🚀' if speedup > 1 else '📉'}")

    return triton_kernel_ms, cuda_kernel_ms, pytorch_kernel_ms
