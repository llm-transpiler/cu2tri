import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 启用同步调试
import torch
import numpy as np
import ctypes
import gc

# 导入MSDeformAttnFunction
from msdeform_version import msdeform_kernel

def reset_cuda_context():
    """重置CUDA上下文"""
    print("🔄 Resetting CUDA context...")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print("✅ CUDA context reset successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to reset CUDA context: {e}")
        return False

def check_cuda_health():
    """检查CUDA健康状态"""
    print("🩺 Checking CUDA health...")
    try:
        # 基本CUDA操作
        device = torch.device("cuda")
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor.T)
        torch.cuda.synchronize()
        
        print(f"✅ Basic CUDA operations working")
        print(f"  - Device: {device}")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"  - Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ CUDA health check failed: {e}")
        return False

def create_minimal_test_data():
    """创建最小化的测试数据"""
    print("📊 Creating minimal test data...")
    torch.manual_seed(42)
    device = "cuda"
    
    # 使用更小的参数进行测试
    lq, m, d, l, k = 2, 2, 4, 2, 2  # 大幅缩小
    
    # 空间形状
    spatial_shapes = [(4, 4), (2, 2)]
    total_spatial = sum(h * w for h, w in spatial_shapes)
    
    print(f"  - Params: lq={lq}, m={m}, d={d}, l={l}, k={k}")
    print(f"  - Spatial shapes: {spatial_shapes}")
    print(f"  - Total spatial: {total_spatial}")
    
    try:
        # 创建张量
        value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device)
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
        level_start_index = torch.tensor([0, 16], dtype=torch.int64, device=device)
        sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device) * 0.8 + 0.1
        attention_weights = torch.rand(lq, m, l*k, dtype=torch.float32, device=device)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        inputs = (value, spatial_shapes_tensor, level_start_index, sampling_locations, attention_weights)
        
        print(f"✅ Minimal test data created successfully")
        print(f"  - value: {value.shape}")
        print(f"  - sampling_locations: {sampling_locations.shape}")
        print(f"  - attention_weights: {attention_weights.shape}")
        
        return inputs
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return None

def test_msdeform_isolated():
    """独立测试MSDeformAttnFunction"""
    print("\n🧪 Testing MSDeformAttnFunction in isolation...")
    
    if not reset_cuda_context():
        return False
        
    if not check_cuda_health():
        return False
    
    inputs = create_minimal_test_data()
    if inputs is None:
        return False
    
    try:
        print("🚀 Running MSDeformAttnFunction...")
        output = msdeform_kernel(*inputs)
        torch.cuda.synchronize()
        
        print(f"✅ MSDeformAttnFunction succeeded!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output mean: {output.mean().item():.6f}")
        print(f"  - Output std: {output.std().item():.6f}")
        
        return True
    except Exception as e:
        print(f"❌ MSDeformAttnFunction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_cuda_kernel_issues():
    """调试CUDA kernel问题"""
    print("\n🔍 Debugging CUDA kernel issues...")
    
    # 检查CUDA kernel文件
    import sys
    sys.path.insert(0, '/workspace/cu2til')
    from cu2til.tools.builder import compile_cuda_kernel, load_cuda_kernel
    
    TESTCASE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    
    try:
        print(f"📂 Testcase root: {TESTCASE_ROOT_DIR}")
        
        # 检查文件是否存在
        kernel_file = os.path.join(TESTCASE_ROOT_DIR, "cuda_", "kernel.cu")
        if os.path.exists(kernel_file):
            print(f"✅ CUDA kernel file exists: {kernel_file}")
        else:
            print(f"❌ CUDA kernel file not found: {kernel_file}")
            return False
        
        # 尝试编译
        print("🔨 Attempting to compile CUDA kernel...")
        compile_cuda_kernel(TESTCASE_ROOT_DIR)
        print("✅ CUDA kernel compilation succeeded")
        
        # 检查输出库文件
        lib_file = os.path.join(TESTCASE_ROOT_DIR, "cuda_", "lib_cuda_kernel.so")
        if os.path.exists(lib_file):
            print(f"✅ Library file exists: {lib_file}")
        else:
            print(f"❌ Library file not found: {lib_file}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CUDA kernel compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_memory_layout():
    """分析内存布局问题"""
    print("\n🧠 Analyzing memory layout...")
    
    inputs = create_minimal_test_data()
    if inputs is None:
        return False
    
    value, spatial_shapes_tensor, level_start_index, sampling_locations, attention_weights = inputs
    
    print("📏 Memory layout analysis:")
    print(f"  - value: {value.shape} = {value.numel()} elements")
    print(f"    - is_contiguous: {value.is_contiguous()}")
    print(f"    - stride: {value.stride()}")
    print(f"    - data_ptr: {hex(value.data_ptr())}")
    
    print(f"  - spatial_shapes: {spatial_shapes_tensor.shape} = {spatial_shapes_tensor.numel()} elements")
    print(f"    - dtype: {spatial_shapes_tensor.dtype}")
    print(f"    - is_contiguous: {spatial_shapes_tensor.is_contiguous()}")
    
    print(f"  - level_start_index: {level_start_index.shape} = {level_start_index.numel()} elements")
    print(f"    - dtype: {level_start_index.dtype}")
    print(f"    - is_contiguous: {level_start_index.is_contiguous()}")
    
    print(f"  - sampling_locations: {sampling_locations.shape} = {sampling_locations.numel()} elements")
    print(f"    - is_contiguous: {sampling_locations.is_contiguous()}")
    
    print(f"  - attention_weights: {attention_weights.shape} = {attention_weights.numel()} elements")
    print(f"    - is_contiguous: {attention_weights.is_contiguous()}")
    
    # 检查内存对齐
    alignment_check = []
    for name, tensor in [
        ("value", value),
        ("spatial_shapes", spatial_shapes_tensor), 
        ("level_start_index", level_start_index),
        ("sampling_locations", sampling_locations),
        ("attention_weights", attention_weights)
    ]:
        ptr = tensor.data_ptr()
        is_aligned = (ptr % 16 == 0)  # 检查16字节对齐
        alignment_check.append((name, is_aligned, ptr))
        print(f"    - {name} alignment (16-byte): {'✅' if is_aligned else '❌'} (ptr: {hex(ptr)})")
    
    return True

def main():
    """主调试函数"""
    print("🐛 CUDA Debug Session")
    print("=" * 50)
    
    # Step 1: 基础健康检查
    if not check_cuda_health():
        print("💥 Basic CUDA operations failed - exiting")
        return
    
    # Step 2: 内存布局分析
    if not analyze_memory_layout():
        print("💥 Memory layout analysis failed")
        return
    
    # Step 3: MSDeformAttnFunction独立测试
    if not test_msdeform_isolated():
        print("💥 MSDeformAttnFunction test failed")
        return
    
    # Step 4: CUDA kernel调试
    if not debug_cuda_kernel_issues():
        print("💥 CUDA kernel debug failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Debug session completed successfully!")
    print("  - All basic tests passed")
    print("  - MSDeformAttnFunction is working")
    print("  - CUDA kernel compilation succeeded")
    print("\n💡 Next steps:")
    print("  - The issue might be in the CUDA kernel implementation itself")
    print("  - Try running the original test case to isolate the problem")
    print("  - Consider using smaller test data to identify the exact issue")

if __name__ == "__main__":
    main()
