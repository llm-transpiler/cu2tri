"""Kernel testing utilities for LLM transpiler."""
import torch
import time
from llm_trans.tools.builder import load_cuda_kernel, load_c_kernel

SEED = 46


def benchmark_gpu_kernel(kernel_func, inputs, warmup=10, iterations=100):
    """GPU performance benchmarking."""
    # Warmup
    for _ in range(warmup):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    # Measure execution time
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_func(*inputs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms

    times.sort()
    return times[len(times) // 2]


def compare_results(output_ref, output_test, atol=1e-2, rtol=1e-2, test_type=["PyTorch", "CUDA"]):
    """Compare two tensor outputs."""
    if output_ref.device != output_test.device:
        output_ref = output_ref.to(output_test.device)

    output_ref = output_ref.to(torch.float32)
    output_test = output_test.to(torch.float32)

    diff = torch.abs(output_ref - output_test)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    rel_err = diff / torch.maximum(torch.abs(output_ref), torch.tensor(1e-9))
    max_rel_err = torch.max(rel_err).item()

    is_close = torch.allclose(output_ref, output_test, atol=atol, rtol=rtol)

    print(f"\n{'='*60}")
    print(f"📊 {test_type[0]} vs {test_type[1]}")
    print(f"{'='*60}")
    print(f"Max Abs Error: {max_diff:.2e}")
    print(f"Mean Abs Error: {mean_diff:.2e}")
    print(f"Max Rel Error: {max_rel_err:.2e}")
    print(f"Result: {'✅ PASSED' if is_close else '❌ FAILED'}")
    print(f"{'='*60}\n")

    return is_close


def cuda_input_tensor_to_ptr(cuda_all_inputs):
    """Convert CUDA tensor inputs to pointers."""
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]


def check_triton_vs_torch(get_cuda_torch_inputs, params, torch_kernel, triton_kernel,
                          output_tensor_transform=lambda x: x, enable_perf=False):
    """Check Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    print(f"🎮 GPU: {torch.cuda.get_device_name()}")

    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    output_torch = torch_kernel(*torch_all_inputs)
    triton_kernel(*triton_all_inputs)
    output_triton = output_tensor_transform(triton_output_tensors[0])

    checkok = compare_results(output_torch, output_triton, test_type=["PyTorch", "Triton"])

    if enable_perf:
        torch_ms = benchmark_gpu_kernel(torch_kernel, torch_all_inputs)
        triton_ms = benchmark_gpu_kernel(triton_kernel, triton_all_inputs)
        print(f"🚀 Performance: PyTorch={torch_ms:.3f}ms, Triton={triton_ms:.3f}ms, Speedup={torch_ms/triton_ms:.2f}x")

    return checkok


def check_cuda_vs_torch(testcase_root_dir, get_cuda_torch_inputs, params, torch_kernel,
                       get_cuda_argtypes, output_tensor_transform=lambda x: x,
                       enable_perf=False, compile_only=False):
    """Check CUDA kernel against PyTorch reference."""
    try:
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(testcase_root_dir, argtypes, force_compile=False)
        print("✅ CUDA kernel loaded")
    except Exception as e:
        print(f"❌ CUDA load failed: {e}")
        return False

    if compile_only:
        return True

    cuda_all_inputs, torch_all_inputs, cuda_output_tensors = get_cuda_torch_inputs(params)
    cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)

    output_torch = torch_kernel(*torch_all_inputs)
    cuda_kernel(*cuda_all_inputs_ptr)
    output_cuda = output_tensor_transform(cuda_output_tensors[0])

    checkok = compare_results(output_torch, output_cuda, test_type=["PyTorch", "CUDA"])

    if enable_perf and checkok:
        torch_ms = benchmark_gpu_kernel(torch_kernel, torch_all_inputs)
        cuda_ms = benchmark_gpu_kernel(cuda_kernel, cuda_all_inputs_ptr)
        print(f"🚀 Performance: PyTorch={torch_ms:.3f}ms, CUDA={cuda_ms:.3f}ms, Speedup={torch_ms/cuda_ms:.2f}x")

    return checkok


def check_triton_vs_torch_dynamic(get_all_cuda_torch_inputs, params, torch_kernel, triton_kernel,
                                  output_tensor_transform=lambda x: x, enable_perf=False):
    """Check Triton vs PyTorch for multiple shapes."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    all_test_data = get_all_cuda_torch_inputs(params)
    overall_success = True

    for shape, (triton_inputs, torch_inputs, triton_outputs) in all_test_data:
        output_torch = torch_kernel(*torch_inputs)
        triton_kernel(*triton_inputs)
        output_triton = output_tensor_transform(triton_outputs[0])

        shape_ok = compare_results(output_torch, output_triton, test_type=["PyTorch", "Triton"])
        if not shape_ok:
            overall_success = False

    return overall_success
