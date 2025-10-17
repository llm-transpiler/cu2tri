import torch
import time
from eval_.common.benchmark import benchmark_kernel
from cu2til.tools.builder import load_cuda_kernel, load_c_kernel

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

def compare_results(output_ref, output_test, atol=1e-2, rtol=1e-2, test_type=["PyTorch", "CUDA"]):
    """Compare results from two implementations"""
    # Ensure both tensors are on the same device for comparison
    if output_ref.device != output_test.device:
        output_ref = output_ref.to(output_test.device)
    
    output_test = output_test.to(torch.float32)
    output_ref = output_ref.to(torch.float32)
    diff = torch.abs(output_ref - output_test)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    rel_err = diff / torch.maximum(torch.abs(output_ref), torch.tensor(1e-9, device=output_ref.device, dtype=torch.float32))
    max_rel_err = torch.max(rel_err)
    mean_rel_err = torch.mean(rel_err)
    
    # Print header
    print("\n" + "="*80)
    print("🔍 TENSOR COMPARISON RESULTS")
    print("="*80)
    
    # Print tensor shape info
    shape_info = f"Tensor Shape: {tuple(output_ref.shape)}, Total Elements: {output_ref.numel():,}"
    print(f"📊 {shape_info}")
    print("-" * 80)
    
    # Print error statistics in a nice table format
    print("📈 ERR STATISTICS:")
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
    torch_vals = [f"{output_ref.flatten()[idx]:.8f}" for idx in abs_err_indices]
    print(f"{test_type[0]:<15} {' '.join([f'{v:>18}' for v in torch_vals])}")
    
    # Print CUDA values
    cuda_vals = [f"{output_test.flatten()[idx]:.8f}" for idx in abs_err_indices]
    print(f"{test_type[1]:<15} {' '.join([f'{v:>18}' for v in cuda_vals])}")
    
    # Print differences
    diffs = [f"{output_ref.flatten()[idx] - output_test.flatten()[idx]:.8f}" for idx in abs_err_indices]
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
    abs_diffs = [f"{torch.abs(output_ref.flatten()[idx] - output_test.flatten()[idx]):.2e}" for idx in rel_err_indices]
    print(f"{'Abs_Err':<15} {' '.join([f'{d:>18}' for d in abs_diffs])}")
    
    # Print relative error values
    rel_errors = [f"{val:.2e}" for val in rel_err_values]
    rel_err_line = f"{'Rel_Err':<15} {' '.join([f'{e:>18}' for e in rel_errors])}"
    print(f"{RED}{rel_err_line}{END}")
    
    # Print PyTorch values
    torch_vals = [f"{output_ref.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{test_type[0]:<15} {' '.join([f'{v:>18}' for v in torch_vals])}")
    
    # Print CUDA values
    cuda_vals = [f"{output_test.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{test_type[1]:<15} {' '.join([f'{v:>18}' for v in cuda_vals])}")
    
    # Print differences
    diffs = [f"{output_ref.flatten()[idx] - output_test.flatten()[idx]:.8f}" for idx in rel_err_indices]
    print(f"{'Diff':<15} {' '.join([f'{d:>18}' for d in diffs])}")
    
    # Print indices
    indices = [f"{idx}" for idx in rel_err_indices]
    print(f"{'Index':<15} {' '.join([f'{idx:>18}' for idx in indices])}")
    
    # Final result
    print("\n" + "="*80)
    is_close = torch.allclose(output_ref, output_test, atol=atol, rtol=rtol)
    if is_close:
        print(f"✅ RESULT: Tensors match within tolerance (atol={atol:.0e}, rtol={rtol:.0e})")
        status = "PASSED"
    else:
        print(f"❌ RESULT: Tensors do NOT match within tolerance (atol={atol:.0e}, rtol={rtol:.0e})")
        status = "FAILED"
    
    print(f"🎯 STATUS: {status}")
    print("="*80 + "\n")
    
    return is_close


def benchmark_cpu_kernel(kernel_func, inputs, warmup=100, iterations=1000):
    """CPU performance benchmarking using time.perf_counter()"""
    # Warmup
    for _ in range(warmup):
        kernel_func(*inputs)
    
    # Measure execution time
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        kernel_func(*inputs)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Return median time
    times.sort()
    median_time = times[len(times) // 2]
    return median_time

def run_performance_test(test_all_inputs_ptr, baseline_all_inputs, test_kernel, baseline_kernel, test_type=["CUDA", "PyTorch"]):
    """Run performance test"""
    # Check if any of the test types involve C (CPU-only), use CPU benchmarking
    if "C" in test_type:
        print(f"🚀 Performance test (CPU):")
        baseline_ms = benchmark_cpu_kernel(baseline_kernel, baseline_all_inputs)
        test_kernel_ms = benchmark_cpu_kernel(test_kernel, test_all_inputs_ptr)
        
        print(f"📊 Performance comparison:")
        print(f"  {test_type[1]:<10}: {baseline_ms:10.7f} ms")
        print(f"  {test_type[0]:<10}: {test_kernel_ms:10.7f} ms")
        
        if test_kernel_ms > 0:
            speedup = baseline_ms / test_kernel_ms
            print(f"  {test_type[0]:<10} vs {test_type[1]:<10} speedup:    {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        
        return baseline_ms, test_kernel_ms
    else:
        print(f"🚀 GPU performance test:")
        baseline_ms = benchmark_kernel(baseline_kernel, baseline_all_inputs)
        test_kernel_ms = benchmark_kernel(test_kernel, test_all_inputs_ptr)

        print(f"📊 GPU performance comparison:")
        print(f"  {test_type[1]:<10}: {baseline_ms:10.7f} ms")
        print(f"  {test_type[0]:<10}: {test_kernel_ms:10.7f} ms")

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
    print(f"  CUDA    : {cuda_kernel_ms:10.7f} ms")
    print(f"  Triton  : {triton_kernel_ms:10.7f} ms")
    print(f"  PyTorch : {pytorch_kernel_ms:10.7f} ms")

    # Calculate and display speedups
    if pytorch_kernel_ms > 0:
        speedup = pytorch_kernel_ms / cuda_kernel_ms
        print(f"  CUDA vs PyTorch   speedup:    {speedup:.3f}x {'🚀' if speedup > 1 else '📉'}")
    
    if triton_kernel_ms > 0:
        speedup = cuda_kernel_ms / triton_kernel_ms
        print(f"  Triton vs CUDA    speedup:    {speedup:.3f}x {'🚀' if speedup > 1 else '📉'}")
        speedup = pytorch_kernel_ms / triton_kernel_ms
        print(f"  Triton vs PyTorch speedup:    {speedup:.3f}x {'🚀' if speedup > 1 else '📉'}")

    return triton_kernel_ms, cuda_kernel_ms, pytorch_kernel_ms

def cuda_input_tensor_to_ptr(cuda_all_inputs):
    return [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in cuda_all_inputs]

def check_triton_vs_torch(get_cuda_torch_inputs, params, torch_kernel, triton_kernel, output_tensor_transform=lambda x: x, enable_perf=False):
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")

    print(f"📋 Creating GPU test data...")
    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    output_torch = torch_kernel(*torch_all_inputs)
    
    print(f"⚡ Running Triton kernel...")
    
    triton_kernel(*triton_all_inputs)

    output_triton = output_tensor_transform(triton_output_tensors[0])
    checkok = compare_results(output_torch, output_triton, atol=1e-2, rtol=1e-2, test_type=["PyTorch", "Triton"])
    if enable_perf:
        run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   Triton  : {output_triton.flatten()[:4]}")
    
    return checkok

def check_triton_vs_torch(get_cuda_torch_inputs, params, torch_kernel, triton_kernel, output_tensor_transform=lambda x: x, enable_perf=False):
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")

    print(f"📋 Creating GPU test data...")
    triton_all_inputs, torch_all_inputs, triton_output_tensors = get_cuda_torch_inputs(params)
    output_torch = torch_kernel(*torch_all_inputs)
    
    print(f"⚡ Running Triton kernel...")
    
    triton_kernel(*triton_all_inputs)

    output_triton = output_tensor_transform(triton_output_tensors[0])
    checkok = compare_results(output_torch, output_triton, atol=1e-2, rtol=1e-2, test_type=["PyTorch", "Triton"])
    if enable_perf:
        run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   Triton  : {output_triton.flatten()[:4]}")
    
    return checkok

def check_cuda_vs_torch(testcase_root_dir, get_cuda_torch_inputs, params, torch_kernel, get_cuda_argtypes, output_tensor_transform=lambda x: x, enable_perf=False, compile_only=False):
    try:
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(testcase_root_dir, argtypes, force_compile=False)
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

    output_cuda = output_tensor_transform(cuda_output_tensors[0])
    checkok = compare_results(output_torch, output_cuda, test_type=["PyTorch", "CUDA"])

    if enable_perf and checkok:
        run_performance_test(cuda_all_inputs_ptr, torch_all_inputs, cuda_kernel, torch_kernel)

    print(f"\n🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   CUDA    : {output_cuda.flatten()[:4]}")

def check_cuda_vs_torch_dynamic(testcase_root_dir, get_all_cuda_torch_inputs, params, torch_kernel, get_cuda_argtypes, output_tensor_transform=lambda x: x, enable_perf=False, compile_only=False):
    """Check CUDA vs PyTorch for multiple shapes dynamically"""
    try:
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(testcase_root_dir, argtypes, force_compile=False)
        print(f"✅ CUDA kernel loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library processing failed: {e}")
        return False

    if compile_only:
        print(f"🔧 Compile-only mode: CUDA kernel compilation completed successfully")
        return True

    # Get all test data for different shapes
    all_test_data = get_all_cuda_torch_inputs(params)
    
    print(f"🧪 Running dynamic shape tests for {len(all_test_data)} different shapes...")
    
    overall_success = True
    successful_shapes = []
    failed_shapes = []
    performance_results = []
    
    for i, (shape, test_data) in enumerate(all_test_data):
        print(f"\n" + "="*100)
        print(f"🔬 TEST CASE {i+1}/{len(all_test_data)}: SHAPE {shape}")
        print(f"📊 Total elements: {torch.prod(torch.tensor(shape)).item():,}")
        print("="*100)
        
        cuda_all_inputs, torch_all_inputs, cuda_output_tensors = test_data
        cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)
        
        try:
            # Run PyTorch reference
            print(f"🔥 Running PyTorch reference...")
            output_torch = torch_kernel(*torch_all_inputs)
            
            # Run CUDA kernel
            print(f"⚡ Running CUDA kernel...")
            cuda_kernel(*cuda_all_inputs_ptr)
            
            output_cuda = output_tensor_transform(cuda_output_tensors[0])
            
            # Compare results
            shape_success = compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2, test_type=["PyTorch", "CUDA"])
            
            if shape_success:
                successful_shapes.append(shape)
                print(f"✅ Shape {shape}: PASSED")
                
                # Performance test for successful cases
                if enable_perf:
                    pytorch_ms, cuda_ms = run_performance_test(cuda_all_inputs_ptr, torch_all_inputs, cuda_kernel, torch_kernel)
                    performance_results.append((shape, pytorch_ms, cuda_ms))
            else:
                failed_shapes.append(shape)
                overall_success = False
                print(f"❌ Shape {shape}: FAILED")
            
            # Display sample results
            print(f"🔬 Sample Output (first 4 values):")
            print(f"   PyTorch : {output_torch.flatten()[:4]}")
            print(f"   CUDA    : {output_cuda.flatten()[:4]}")
            
        except Exception as e:
            failed_shapes.append(shape)
            overall_success = False
            print(f"❌ Shape {shape}: ERROR - {e}")
    
    # Summary report
    print(f"\n" + "="*100)
    print(f"📋 DYNAMIC SHAPE TEST SUMMARY")
    print("="*100)
    print(f"Total test cases: {len(all_test_data)}")
    print(f"✅ Successful: {len(successful_shapes)}")
    print(f"❌ Failed: {len(failed_shapes)}")
    print(f"🎯 Success rate: {len(successful_shapes)/len(all_test_data)*100:.1f}%")
    
    if successful_shapes:
        print(f"\n✅ Successful shapes:")
        for shape in successful_shapes:
            elements = torch.prod(torch.tensor(shape)).item()
            print(f"   {shape} ({elements:,} elements)")
    
    if failed_shapes:
        print(f"\n❌ Failed shapes:")
        for shape in failed_shapes:
            elements = torch.prod(torch.tensor(shape)).item()
            print(f"   {shape} ({elements:,} elements)")
    
    # Performance summary
    if enable_perf and performance_results:
        print(f"\n🚀 PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Shape':<20} {'Elements':<12} {'PyTorch(ms)':<15} {'CUDA(ms)':<15} {'Speedup':<10}")
        print("-" * 80)
        for shape, pytorch_ms, cuda_ms in performance_results:
            elements = torch.prod(torch.tensor(shape)).item()
            speedup = pytorch_ms / cuda_ms if cuda_ms > 0 else 0
            speedup_icon = "🚀" if speedup > 1 else "📉"
            print(f"{str(shape):<20} {elements:<12,} {pytorch_ms:<15.7f} {cuda_ms:<15.7f} {speedup:<7.2f}x {speedup_icon}")
    
    print("="*100)
    final_status = "PASSED" if overall_success else "FAILED"
    status_icon = "✅" if overall_success else "❌"
    print(f"{status_icon} OVERALL STATUS: {final_status}")
    print("="*100 + "\n")
    
    return overall_success

def check_triton_vs_torch_dynamic(get_all_cuda_torch_inputs, params, torch_kernel, triton_kernel, output_tensor_transform=lambda x: x, enable_perf=False):
    """Check Triton vs PyTorch for multiple shapes dynamically"""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")

    # Get all test data for different shapes
    all_test_data = get_all_cuda_torch_inputs(params)
    
    print(f"🧪 Running dynamic shape tests for {len(all_test_data)} different shapes...")
    
    overall_success = True
    successful_shapes = []
    failed_shapes = []
    performance_results = []
    
    for i, (shape, test_data) in enumerate(all_test_data):
        print(f"\n" + "="*100)
        print(f"🔬 TEST CASE {i+1}/{len(all_test_data)}: SHAPE {shape}")
        print(f"📊 Total elements: {torch.prod(torch.tensor(shape)).item():,}")
        print("="*100)
        
        triton_all_inputs, torch_all_inputs, triton_output_tensors = test_data
        
        try:
            # Run PyTorch reference
            print(f"🔥 Running PyTorch reference...")
            output_torch = torch_kernel(*torch_all_inputs)
            
            # Run Triton kernel
            print(f"⚡ Running Triton kernel...")
            triton_kernel(*triton_all_inputs)
            
            output_triton = output_tensor_transform(triton_output_tensors[0])
            
            # Compare results
            shape_success = compare_results(output_torch, output_triton, atol=1e-1, rtol=1e-2, test_type=["PyTorch", "Triton"])
            
            if shape_success:
                successful_shapes.append(shape)
                print(f"✅ Shape {shape}: PASSED")
                
                # Performance test for successful cases
                if enable_perf:
                    pytorch_ms, triton_ms = run_performance_test(triton_all_inputs, torch_all_inputs, triton_kernel, torch_kernel, test_type=["Triton", "PyTorch"])
                    performance_results.append((shape, pytorch_ms, triton_ms))
            else:
                failed_shapes.append(shape)
                overall_success = False
                print(f"❌ Shape {shape}: FAILED")
            
            # Display sample results
            print(f"🔬 Sample Output (first 4 values):")
            print(f"   PyTorch : {output_torch.flatten()[:4]}")
            print(f"   Triton  : {output_triton.flatten()[:4]}")
            
        except Exception as e:
            failed_shapes.append(shape)
            overall_success = False
            print(f"❌ Shape {shape}: ERROR - {e}")
    
    # Summary report
    print(f"\n" + "="*100)
    print(f"📋 DYNAMIC SHAPE TEST SUMMARY")
    print("="*100)
    print(f"Total test cases: {len(all_test_data)}")
    print(f"✅ Successful: {len(successful_shapes)}")
    print(f"❌ Failed: {len(failed_shapes)}")
    print(f"🎯 Success rate: {len(successful_shapes)/len(all_test_data)*100:.1f}%")
    
    if successful_shapes:
        print(f"\n✅ Successful shapes:")
        for shape in successful_shapes:
            elements = torch.prod(torch.tensor(shape)).item()
            print(f"   {shape} ({elements:,} elements)")
    
    if failed_shapes:
        print(f"\n❌ Failed shapes:")
        for shape in failed_shapes:
            elements = torch.prod(torch.tensor(shape)).item()
            print(f"   {shape} ({elements:,} elements)")
    
    # Performance summary
    if enable_perf and performance_results:
        print(f"\n🚀 PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Shape':<20} {'Elements':<12} {'PyTorch(ms)':<15} {'Triton(ms)':<15} {'Speedup':<10}")
        for shape, pytorch_ms, triton_ms in performance_results:
            print("-" * 80)
            elements = torch.prod(torch.tensor(shape)).item()
            speedup = pytorch_ms / triton_ms if triton_ms > 0 else 0
            speedup_icon = "🚀" if speedup > 1 else "📉"
            print(f"{str(shape):<20} {elements:<12,} {pytorch_ms:<15.7f} {triton_ms:<15.7f} {speedup:<7.2f}x {speedup_icon}")
    
    print("="*100)
    final_status = "PASSED" if overall_success else "FAILED"
    status_icon = "✅" if overall_success else "❌"
    print(f"{status_icon} OVERALL STATUS: {final_status}")
    print("="*100 + "\n")
    
    return overall_success

def check_c_vs_torch(testcase_root_dir, get_c_torch_inputs, params, torch_kernel, get_c_argtypes, output_tensor_transform=lambda x: x, enable_perf=False, compile_only=False):
    """Check C implementation vs PyTorch reference"""
    try:
        argtypes = get_c_argtypes()
        c_kernel = load_c_kernel(testcase_root_dir, argtypes, force_compile=False)
        print(f"✅ C kernel loaded successfully")
    except Exception as e:
        print(f"❌ C library processing failed: {e}")
        return

    if compile_only:
        print(f"🔧 Compile-only mode: C kernel compilation completed successfully")
        return

    print(f"📋 Creating CPU test data...")
    c_all_inputs, torch_all_inputs, c_output_tensors = get_c_torch_inputs(params)

    # Run PyTorch reference
    output_torch = torch_kernel(*torch_all_inputs)

    print(f"⚡ Running C kernel...")
    c_kernel(*c_all_inputs)

    output_c = output_tensor_transform(c_output_tensors[0])
    checkok = compare_results(output_torch, output_c, test_type=["PyTorch", "C"])

    if enable_perf and checkok:
        run_performance_test(c_all_inputs, torch_all_inputs, c_kernel, torch_kernel, test_type=["C", "PyTorch"])

    print(f"\n🔬 Sample Output (first 4 values):")
    print(f"   PyTorch : {output_torch.flatten()[:4]}")
    print(f"   C       : {output_c.flatten()[:4]}")
    
    return checkok

def check_triton_vs_c(testcase_root_dir, get_c_torch_inputs, params, triton_kernel, get_c_argtypes, triton_output_transform=lambda x: x, c_output_transform=lambda x: x, enable_perf=False, compile_only=False):
    """Check Triton implementation vs C implementation"""
    
    # Check GPU availability for Triton
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Load C kernel
    try:
        argtypes = get_c_argtypes()
        c_kernel = load_c_kernel(testcase_root_dir, argtypes, force_compile=False)
        print(f"✅ C kernel loaded successfully")
    except Exception as e:
        print(f"❌ C library processing failed: {e}")
        return False

    if compile_only:
        print(f"🔧 Compile-only mode: C kernel compilation completed successfully")
        return True

    # Generate test data from same source (CPU)
    print(f"📋 Creating test data...")
    
    # Get C inputs (CPU) - this generates the base data
    c_all_inputs, torch_all_inputs, c_output_tensors = get_c_torch_inputs(params)
    
    # For Triton, we need to convert the PyTorch tensors to GPU and create GPU output tensor
    A_gpu = torch_all_inputs[0].cuda()
    B_gpu = torch_all_inputs[1].cuda()
    output_gpu = torch.empty_like(A_gpu)
    triton_all_inputs = [A_gpu, B_gpu, output_gpu, params.total_elements]
    triton_output_tensors = [output_gpu]

    # Run Triton kernel
    print(f"⚡ Running Triton kernel...")
    triton_kernel(*triton_all_inputs)
    output_triton = triton_output_transform(triton_output_tensors[0])
    
    # Run C kernel
    print(f"⚡ Running C kernel...")
    c_kernel(*c_all_inputs)
    output_c = c_output_transform(c_output_tensors[0])
    
    # Convert outputs to same device for comparison (use CPU for consistency)
    if output_triton.device.type == 'cuda':
        output_triton = output_triton.cpu()
    
    # Compare results
    checkok = compare_results(output_triton, output_c, test_type=["Triton", "C"])
    
    if enable_perf and checkok:
        # Note: Performance comparison between GPU and CPU may not be meaningful
        print(f"⚠️  Performance comparison between GPU (Triton) and CPU (C) may not be meaningful")
        # You could add performance comparison here if needed
    
    # Display sample results
    print(f"🔬 Sample Output (first 4 values):")
    print(f"   Triton  : {output_triton.flatten()[:4]}")
    print(f"   C       : {output_c.flatten()[:4]}")
    
    return checkok

def check_all_kernels(testcase_root_dir, get_cuda_torch_inputs, params, torch_kernel, triton_kernel, get_cuda_argtypes, output_tensor_transform=lambda x: x, enable_perf=False, compile_only=False):
    """Check all kernels: CUDA vs PyTorch, Triton vs PyTorch, and performance comparison"""
    
    print("🚀 Starting comprehensive kernel testing...")
    print("="*80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, please ensure there is a GPU environment")
        return False
    
    device = torch.cuda.current_device()
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Test results tracking
    cuda_success = False
    triton_success = False
    
    # 1. Test CUDA vs PyTorch
    print(f"\n" + "="*80)
    print("🔧 PHASE 1: CUDA vs PyTorch Testing")
    print("="*80)
    
    try:
        argtypes = get_cuda_argtypes()
        cuda_kernel = load_cuda_kernel(testcase_root_dir, argtypes, force_compile=False)
        print(f"✅ CUDA kernel loaded successfully")
        
        if not compile_only:
            print(f"📋 Creating CUDA test data...")
            cuda_all_inputs, torch_all_inputs, cuda_output_tensors = get_cuda_torch_inputs(params)
            cuda_all_inputs_ptr = cuda_input_tensor_to_ptr(cuda_all_inputs)
            
            # Run PyTorch reference
            print(f"🔥 Running PyTorch reference...")
            output_torch = torch_kernel(*torch_all_inputs)
            
            # Run CUDA kernel
            print(f"⚡ Running CUDA kernel...")
            cuda_kernel(*cuda_all_inputs_ptr)
            
            output_cuda = output_tensor_transform(cuda_output_tensors[0])
            cuda_success = compare_results(output_torch, output_cuda, test_type=["PyTorch", "CUDA"])
            
            if cuda_success:
                print(f"✅ CUDA vs PyTorch: PASSED")
            else:
                print(f"❌ CUDA vs PyTorch: FAILED")
                
        else:
            print(f"🔧 Compile-only mode: CUDA kernel compilation completed")
            cuda_success = True
            
    except Exception as e:
        print(f"❌ CUDA testing failed: {e}")
        cuda_success = False
    
    # 2. Test Triton vs CUDA
    print(f"\n" + "="*80) 
    print("⚡ PHASE 2: Triton vs CUDA Testing")
    print("="*80)
    
    if not compile_only:
        if cuda_success:
            try:
                print(f"📋 Creating Triton test data...")
                triton_all_inputs, torch_all_inputs_triton, triton_output_tensors = get_cuda_torch_inputs(params)
                
                # Run Triton kernel
                print(f"⚡ Running Triton kernel...")
                triton_kernel(*triton_all_inputs)
                
                output_triton = output_tensor_transform(triton_output_tensors[0])
                triton_success = compare_results(output_cuda, output_triton, test_type=["CUDA", "Triton"])
                
                if triton_success:
                    print(f"✅ Triton vs CUDA: PASSED")
                else:
                    print(f"❌ Triton vs CUDA: FAILED")
                    
            except Exception as e:
                print(f"❌ Triton testing failed: {e}")
                triton_success = False
        else:
            print(f"⚠️  Skipping Triton vs CUDA test: CUDA test failed")
            triton_success = False
    else:
        print(f"🔧 Compile-only mode: Skipping Triton testing")
        triton_success = True
    
    # 3. Performance comparison (if all tests passed and performance is enabled)
    if enable_perf and not compile_only and cuda_success and triton_success:
        print(f"\n" + "="*80)
        print("🚀 PHASE 3: Performance Comparison")
        print("="*80)
        
        try:
            # Prepare inputs for all kernels
            triton_ms, cuda_ms, pytorch_ms = run_performance_all(
                triton_all_inputs, cuda_all_inputs_ptr, torch_all_inputs,
                triton_kernel, cuda_kernel, torch_kernel
            )
        except Exception as e:
            print(f"❌ Performance testing failed: {e}")
    
    # 4. Final Summary
    print(f"\n" + "="*80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    if not compile_only:
        # Sample output comparison
        if cuda_success and triton_success:
            print(f"🔬 Sample Output Comparison (first 4 values):")
            print(f"   PyTorch : {output_torch.flatten()[:4]}")
            if cuda_success:
                print(f"   CUDA    : {output_cuda.flatten()[:4]}")
            if triton_success:
                print(f"   Triton  : {output_triton.flatten()[:4]}")
        
        print(f"\nTest Results:")
        print(f"  Triton vs CUDA    : {'✅ PASSED' if triton_success else '❌ FAILED'}")
        print(f"  CUDA   vs PyTorch : {'✅ PASSED' if cuda_success else '❌ FAILED'}")
        
        overall_success = cuda_success and triton_success
        print(f"\n🎯 OVERALL STATUS: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    else:
        print(f"🔧 Compile-only mode completed")
        overall_success = cuda_success
    
    print("="*80)
    
    return overall_success if not compile_only else True

