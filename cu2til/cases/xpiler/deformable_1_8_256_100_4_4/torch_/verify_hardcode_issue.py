import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import ctypes
import gc

# 导入MSDeformAttnFunction 
from msdeform_version import msdeform_kernel
import sys
sys.path.insert(0, '/workspace/cu2util')
from cu2til.tools.builder import compile_cuda_kernel, load_cuda_kernel

TESTCASE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

def reset_cuda():
    """重置CUDA状态"""
    torch.cuda.empty_cache()
    gc.collect() 
    torch.cuda.synchronize()

def test_exact_hardcoded_params():
    """测试完全匹配硬编码参数的情况"""
    print("🎯 Testing with EXACT hardcoded parameters")
    print("   CUDA kernel expects: n=1, lq=100, m=8, d=256, l=4")
    
    reset_cuda()
    torch.manual_seed(42)
    device = "cuda"
    
    # 使用完全匹配CUDA kernel硬编码的参数
    n, lq, m, d, l, k = 1, 100, 8, 256, 4, 4
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]  # 原始spatial shapes
    
    try:
        # 计算level starts
        total_spatial = sum(h * w for h, w in spatial_shapes)
        level_starts = []
        current_start = 0
        for h, w in spatial_shapes:
            level_starts.append(current_start)
            current_start += h * w
        
        print(f"   - Spatial shapes: {spatial_shapes}")
        print(f"   - Total spatial: {total_spatial}")
        print(f"   - Level starts: {level_starts}")
        
        # 创建数据
        value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device).normal_(0.0, 0.5)
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
        level_start_index = torch.tensor(level_starts, dtype=torch.int64, device=device)
        sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
        sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
        attention_weights = torch.rand(lq, m, l*k, dtype=torch.float32, device=device)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        inputs = (value, spatial_shapes_tensor, level_start_index, sampling_locations, attention_weights)
        
        # 测试MSDeformAttnFunction
        print("   🚀 Testing MSDeformAttnFunction...")
        msdeform_output = msdeform_kernel(*inputs)
        torch.cuda.synchronize()
        print(f"   ✅ MSDeformAttnFunction: SUCCESS - {msdeform_output.shape}")
        
        # 测试CUDA kernel
        print("   🚀 Testing CUDA Kernel...")
        compile_cuda_kernel(TESTCASE_ROOT_DIR)
        argtypes = [ctypes.c_void_p] * 6
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes)
        
        # 创建输出张量
        cuda_output = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
        
        # 准备指针
        value_ptr = value.contiguous().data_ptr()
        value_spatial_shapes_ptr = spatial_shapes_tensor.int().contiguous().data_ptr()
        level_start_index_ptr = level_start_index.int().contiguous().data_ptr()
        sampling_locations_ptr = sampling_locations.contiguous().data_ptr()
        attention_weights_ptr = attention_weights.contiguous().data_ptr()
        output_ptr = cuda_output.contiguous().data_ptr()
        
        # 调用CUDA kernel
        cuda_kernel(
            ctypes.c_void_p(value_ptr),
            ctypes.c_void_p(value_spatial_shapes_ptr),
            ctypes.c_void_p(level_start_index_ptr),
            ctypes.c_void_p(sampling_locations_ptr),
            ctypes.c_void_p(attention_weights_ptr),
            ctypes.c_void_p(output_ptr)
        )
        
        torch.cuda.synchronize()
        print(f"   ✅ CUDA Kernel: SUCCESS - {cuda_output.shape}")
        
        # 比较结果
        diff = torch.abs(msdeform_output - cuda_output).max().item()
        print(f"   📊 Max difference: {diff:.2e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

def test_mismatched_params():
    """测试参数不匹配的情况"""
    print("\n🚫 Testing with MISMATCHED parameters")
    print("   Using smaller params while CUDA kernel expects hardcoded values")
    
    reset_cuda()
    torch.manual_seed(42)
    device = "cuda"
    
    # 使用不匹配的较小参数（这应该失败）
    lq, m, d, l, k = 20, 4, 32, 4, 4
    spatial_shapes = [(8, 8), (4, 4), (2, 2), (1, 1)]
    
    try:
        # 计算level starts
        total_spatial = sum(h * w for h, w in spatial_shapes)
        level_starts = []
        current_start = 0
        for h, w in spatial_shapes:
            level_starts.append(current_start)
            current_start += h * w
        
        print(f"   - Using: lq={lq}, m={m}, d={d}, l={l}, k={k}")
        print(f"   - Spatial shapes: {spatial_shapes}")
        print(f"   - Total spatial: {total_spatial}")
        
        # 创建数据
        value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device).normal_(0.0, 0.5)
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
        level_start_index = torch.tensor(level_starts, dtype=torch.int64, device=device)
        sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
        sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
        attention_weights = torch.rand(lq, m, l*k, dtype=torch.float32, device=device)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        inputs = (value, spatial_shapes_tensor, level_start_index, sampling_locations, attention_weights)
        
        # 测试MSDeformAttnFunction（这应该工作）
        print("   🚀 Testing MSDeformAttnFunction...")
        msdeform_output = msdeform_kernel(*inputs)
        torch.cuda.synchronize()
        print(f"   ✅ MSDeformAttnFunction: SUCCESS - {msdeform_output.shape}")
        
        # 测试CUDA kernel（这应该失败）
        print("   🚀 Testing CUDA Kernel...")
        compile_cuda_kernel(TESTCASE_ROOT_DIR)
        argtypes = [ctypes.c_void_p] * 6
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes)
        
        # 创建输出张量 - 注意：这里我们仍然需要使用硬编码的大小！
        cuda_output = torch.empty(100, 8, 256, dtype=torch.float32, device="cuda")  # 硬编码大小
        
        # 准备指针
        value_ptr = value.contiguous().data_ptr()
        value_spatial_shapes_ptr = spatial_shapes_tensor.int().contiguous().data_ptr()
        level_start_index_ptr = level_start_index.int().contiguous().data_ptr()
        sampling_locations_ptr = sampling_locations.contiguous().data_ptr()
        attention_weights_ptr = attention_weights.contiguous().data_ptr()
        output_ptr = cuda_output.contiguous().data_ptr()
        
        print("   ⚠️  Calling CUDA kernel with mismatched data sizes...")
        print("   ⚠️  Kernel expects: lq=100, m=8, d=256")
        print(f"   ⚠️  But data is:    lq={lq}, m={m}, d={d}")
        
        # 调用CUDA kernel
        cuda_kernel(
            ctypes.c_void_p(value_ptr),
            ctypes.c_void_p(value_spatial_shapes_ptr),
            ctypes.c_void_p(level_start_index_ptr),
            ctypes.c_void_p(sampling_locations_ptr),
            ctypes.c_void_p(attention_weights_ptr),
            ctypes.c_void_p(output_ptr)
        )
        
        torch.cuda.synchronize()
        print(f"   🤔 Unexpected: CUDA Kernel didn't fail - {cuda_output.shape}")
        return True
        
    except Exception as e:
        print(f"   ✅ Expected failure: {e}")
        return False

def main():
    """主验证函数"""
    print("🔍 Verifying Hardcoded Parameter Issue")
    print("=" * 60)
    
    print("This test verifies that the CUDA kernel fails due to hardcoded parameters")
    print("that don't match the actual input data sizes.")
    
    # 测试1：使用完全匹配的参数
    exact_match_success = test_exact_hardcoded_params()
    
    # 测试2：使用不匹配的参数
    mismatch_failed_as_expected = not test_mismatched_params()
    
    print("\n" + "=" * 60)
    print("📋 Verification Results:")
    print(f"   - Exact match test: {'✅ PASSED' if exact_match_success else '❌ FAILED'}")
    print(f"   - Mismatch test: {'✅ FAILED AS EXPECTED' if mismatch_failed_as_expected else '❌ UNEXPECTED RESULT'}")
    
    if exact_match_success and mismatch_failed_as_expected:
        print("\n🎉 HYPOTHESIS CONFIRMED!")
        print("   The CUDA kernel fails due to hardcoded parameters!")
        print("\n💡 Solution:")
        print("   - MSDeformAttnFunction works with any input size")
        print("   - CUDA kernel only works with lq=100, m=8, d=256, l=4")
        print("   - Recommend using MSDeformAttnFunction instead")
    else:
        print("\n🤔 Results don't match hypothesis, need further investigation")

if __name__ == "__main__":
    main()
