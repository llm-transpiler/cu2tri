import os
import sys
import argparse
from get_data import get_all_cuda_torch_inputs, Params, get_cuda_argtypes # type: ignore
import torch
try:
    from get_data import cuda_output_tensor_transform # type: ignore
except ImportError:
    cuda_output_tensor_transform = lambda x: x

from torch_.ref import flash_attn_kernel # type: ignore
# from cu2til.tools.checker import check_cuda_vs_torch_dynamic
from cu2til.tools.checker import load_cuda_kernel, cuda_input_tensor_to_ptr, compare_results, run_performance_test

TESTCASE_ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, TESTCASE_ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic CUDA kernel test for multiple shapes')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the CUDA kernel without running tests (default: False)')
    parser.add_argument('--no-perf', action='store_true',
                        help='Disable performance testing (default: False, performance testing enabled)')
    parser.add_argument('--shapes', nargs='+', type=int, 
                        help='Custom shapes to test (e.g., --shapes 512 1024 2048). If not provided, uses default test shapes.')
    return parser.parse_args()

def check_cuda_vs_torch_dynamic(testcase_root_dir, get_all_cuda_torch_inputs, params, flash_attn_kernel, get_cuda_argtypes, output_tensor_transform=lambda x: x, enable_perf=False, compile_only=False):
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
            # cuda_st = [t for t in torch_all_inputs if isinstance(t, torch.Tensor)]
            
            # Run CUDA kernel
            print(f"⚡ Running CUDA kernel...")
            cuda_kernel(*cuda_all_inputs_ptr)
            
            output_cuda = output_tensor_transform(cuda_output_tensors[0])
            
            # Run PyTorch reference
            print(f"🔥 Running PyTorch reference...")
            torch_all_inputs = [tensor.transpose(1, 2).contiguous() if isinstance(tensor, torch.Tensor) else tensor for tensor in torch_all_inputs]
            output_torch = flash_attn_kernel(*torch_all_inputs).transpose(1, 2).contiguous()
            # Compare results
            shape_success = compare_results(output_torch, output_cuda, atol=1e-2, rtol=1e-2, test_type=["FA2", "CUDA"])
            
            if shape_success:
                successful_shapes.append(shape)
                print(f"✅ Shape {shape}: PASSED")
                
                # Performance test for successful cases
                if enable_perf:
                    pytorch_ms, cuda_ms = run_performance_test(cuda_all_inputs_ptr, torch_all_inputs, cuda_kernel, flash_attn_kernel)
                    performance_results.append((shape, pytorch_ms, cuda_ms))
            else:
                failed_shapes.append(shape)
                overall_success = False
                print(f"❌ Shape {shape}: FAILED")
            
            # Display sample results
            print(f"🔬 Sample Output (first 4 values):")
            print(f"   FA2     : {output_torch.flatten()[:4]}")
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

if __name__ == "__main__":
    args = parse_args()
    enable_perf = not args.no_perf  # Default is True, disable with --no-perf
    compile_only = args.compile_only  # Default is False
    
    params = Params()
    
    success = check_cuda_vs_torch_dynamic(
        testcase_root_dir=TESTCASE_ROOT_DIR,
        get_all_cuda_torch_inputs=get_all_cuda_torch_inputs,
        params=params,
        flash_attn_kernel=flash_attn_kernel,
        get_cuda_argtypes=get_cuda_argtypes,
        output_tensor_transform=cuda_output_tensor_transform,
        enable_perf=enable_perf,
        compile_only=compile_only
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
