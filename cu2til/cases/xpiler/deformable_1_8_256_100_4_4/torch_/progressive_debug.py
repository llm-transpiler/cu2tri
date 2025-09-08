import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步调试
import torch
import numpy as np
import ctypes
import gc

# 导入必要模块
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

def test_msdeform_scale(lq, m, d, l, k, spatial_shapes):
    """测试MSDeformAttnFunction在特定规模下的表现"""
    print(f"\n🧪 Testing MSDeformAttnFunction: lq={lq}, m={m}, d={d}, l={l}, k={k}")
    
    reset_cuda()
    torch.manual_seed(42)
    device = "cuda"
    
    try:
        # 计算总空间大小
        total_spatial = sum(h * w for h, w in spatial_shapes)
        level_starts = []
        current_start = 0
        for h, w in spatial_shapes:
            level_starts.append(current_start)
            current_start += h * w
        
        print(f"  - Spatial shapes: {spatial_shapes}")
        print(f"  - Total spatial: {total_spatial}")
        print(f"  - Level starts: {level_starts}")
        
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
        output = msdeform_kernel(*inputs)
        torch.cuda.synchronize()
        
        print(f"  ✅ MSDeformAttnFunction: SUCCESS - Output shape: {output.shape}")
        return True, inputs
        
    except Exception as e:
        print(f"  ❌ MSDeformAttnFunction: FAILED - {e}")
        return False, None

def test_cuda_kernel_scale(inputs):
    """测试CUDA kernel在特定输入下的表现"""
    print(f"  🚀 Testing CUDA Kernel...")
    
    if inputs is None:
        print(f"  ❌ CUDA Kernel: SKIPPED - No valid inputs")
        return False
    
    try:
        # 编译和加载CUDA kernel
        compile_cuda_kernel(TESTCASE_ROOT_DIR)
        argtypes = [ctypes.c_void_p] * 6
        cuda_kernel = load_cuda_kernel(TESTCASE_ROOT_DIR, argtypes)
        
        value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = inputs
        
        # 获取参数
        lq, m, lk, _ = sampling_locations.shape
        d = value.shape[2]
        
        # 创建输出张量
        output_gpu = torch.empty(lq, m, d, dtype=torch.float32, device="cuda")
        
        # 准备指针 - 确保数据类型正确
        value_ptr = value.contiguous().data_ptr()
        value_spatial_shapes_ptr = value_spatial_shapes.int().contiguous().data_ptr()  # 转为int32
        level_start_index_ptr = level_start_index.int().contiguous().data_ptr()  # 转为int32
        sampling_locations_ptr = sampling_locations.contiguous().data_ptr()
        attention_weights_ptr = attention_weights.contiguous().data_ptr()
        output_ptr = output_gpu.contiguous().data_ptr()
        
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
        print(f"  ✅ CUDA Kernel: SUCCESS - Output shape: {output_gpu.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ CUDA Kernel: FAILED - {e}")
        return False

def progressive_test():
    """渐进式测试，从小到大"""
    print("🔍 Progressive Scaling Test")
    print("=" * 60)
    
    # 测试用例：从小到大
    test_cases = [
        # (lq, m, d, l, k, spatial_shapes)
        (2, 2, 4, 2, 2, [(4, 4), (2, 2)]),           # 最小
        (4, 2, 8, 2, 2, [(4, 4), (2, 2)]),           # 稍大
        (10, 4, 16, 2, 2, [(8, 8), (4, 4)]),         # 中等
        (20, 4, 32, 4, 4, [(8, 8), (4, 4), (2, 2), (1, 1)]),  # 较大
        (50, 8, 64, 4, 4, [(16, 16), (8, 8), (4, 4), (2, 2)]), # 更大
        (100, 8, 128, 4, 4, [(16, 16), (8, 8), (4, 4), (2, 2)]), # 接近原始
        (100, 8, 256, 4, 4, [(32, 32), (16, 16), (8, 8), (4, 4)]), # 原始大小
    ]
    
    results = []
    
    for i, (lq, m, d, l, k, spatial_shapes) in enumerate(test_cases):
        print(f"\n{'='*20} Test Case {i+1}/{len(test_cases)} {'='*20}")
        
        # 测试MSDeformAttnFunction
        msdeform_success, inputs = test_msdeform_scale(lq, m, d, l, k, spatial_shapes)
        
        # 测试CUDA kernel（只有在MSDeform成功时才测试，避免污染GPU状态）
        cuda_success = False
        if msdeform_success:
            cuda_success = test_cuda_kernel_scale(inputs)
        
        results.append({
            'case': f"lq={lq}, m={m}, d={d}, l={l}, k={k}",
            'spatial_shapes': spatial_shapes,
            'msdeform': msdeform_success,
            'cuda': cuda_success
        })
        
        # 如果CUDA kernel失败了，就不再继续测试更大的case
        if not cuda_success and msdeform_success:
            print(f"\n⚠️ CUDA Kernel failed at this scale, stopping further tests")
            break
        
        # 重置GPU状态防止累积错误
        reset_cuda()
    
    return results

def analyze_results(results):
    """分析测试结果"""
    print(f"\n" + "="*60)
    print("📊 Test Results Analysis")
    print("="*60)
    
    print(f"\n{'Case':<40} {'MSDeform':<12} {'CUDA':<12}")
    print("-" * 65)
    
    for result in results:
        case = result['case']
        msdeform_status = "✅ PASS" if result['msdeform'] else "❌ FAIL"
        cuda_status = "✅ PASS" if result['cuda'] else "❌ FAIL"
        print(f"{case:<40} {msdeform_status:<12} {cuda_status:<12}")
    
    # 找到失败的临界点
    cuda_failures = [r for r in results if not r['cuda'] and r['msdeform']]
    if cuda_failures:
        print(f"\n🎯 CUDA Kernel first failed at:")
        failed_case = cuda_failures[0]
        print(f"   {failed_case['case']}")
        print(f"   Spatial shapes: {failed_case['spatial_shapes']}")
        
        # 计算数据量
        total_spatial = sum(h * w for h, w in failed_case['spatial_shapes'])
        lq = int(failed_case['case'].split('lq=')[1].split(',')[0])
        m = int(failed_case['case'].split('m=')[1].split(',')[0])
        d = int(failed_case['case'].split('d=')[1].split(',')[0])
        l = len(failed_case['spatial_shapes'])
        k = int(failed_case['case'].split('k=')[1])
        
        total_memory_mb = (total_spatial * m * d * 4 + lq * m * l * k * 2 * 4 + lq * m * l * k * 4) / 1024 / 1024
        
        print(f"\n📈 Scale analysis at failure point:")
        print(f"   - Total spatial size: {total_spatial}")
        print(f"   - Estimated memory usage: {total_memory_mb:.2f} MB")
        print(f"   - Value tensor size: {total_spatial * m * d} elements")
        print(f"   - Sampling locations: {lq * m * l * k * 2} elements")
        print(f"   - Attention weights: {lq * m * l * k} elements")
    
    msdeform_success_count = sum(1 for r in results if r['msdeform'])
    cuda_success_count = sum(1 for r in results if r['cuda'])
    
    print(f"\n📋 Summary:")
    print(f"   - MSDeformAttnFunction: {msdeform_success_count}/{len(results)} tests passed")
    print(f"   - CUDA Kernel: {cuda_success_count}/{len(results)} tests passed")
    
    if msdeform_success_count == len(results):
        print(f"   🎉 MSDeformAttnFunction works perfectly at all scales!")
    
    if cuda_success_count == 0:
        print(f"   ⚠️  CUDA Kernel has fundamental issues")
    elif cuda_success_count < len(results):
        print(f"   ⚠️  CUDA Kernel fails at larger scales")

def main():
    """主函数"""
    print("🚀 Progressive Debug Analysis")
    print("Finding the exact scale where CUDA kernel fails")
    print("=" * 60)
    
    results = progressive_test()
    analyze_results(results)

if __name__ == "__main__":
    main()
