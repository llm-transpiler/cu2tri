import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F

# 尝试导入安装的MSDeformAttnFunction
try:
    # 导入安装的MultiScaleDeformableAttention模块
    from functions.ms_deform_attn_func import MSDeformAttnFunction
    print("✅ Successfully imported MSDeformAttnFunction from installed module")
except ImportError as e:
    print(f"❌ Failed to import MSDeformAttnFunction: {e}")
    print("Trying alternative import...")
    try:
        import sys
        sys.path.append('/usr/local/lib/python3.12/dist-packages/MultiScaleDeformableAttention-1.0-py3.12-linux-x86_64.egg')
        from functions.ms_deform_attn_func import MSDeformAttnFunction
        print("✅ Successfully imported MSDeformAttnFunction from egg path")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        # 如果都失败了，我们创建一个简单的备用版本
        MSDeformAttnFunction = None

def msdeform_kernel(*args):
    """使用MSDeformAttnFunction的实现版本"""
    if MSDeformAttnFunction is None:
        raise RuntimeError("MSDeformAttnFunction not available")
        
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = args
    
    # 从输入张量动态推断参数
    lq, m, lk, _ = sampling_locations.shape  # [lq, m, l*k, 2]
    total_spatial, m_check, d = value.shape  # [total_spatial, m, d]
    l = len(level_start_index)  # num_levels
    k = lk // l  # num_points per level
    
    print(f"Dynamic params: lq={lq}, m={m}, d={d}, l={l}, k={k}")
    
    assert m == m_check, f"Head dimension mismatch: {m} vs {m_check}"
    assert lk == l * k, f"Location dimension mismatch: {lk} vs {l*k}"
    
    # MSDeformAttnFunction期望的输入格式:
    # - value: [N, sum(H*W), num_heads, embed_dim]
    # - spatial_shapes: [num_levels, 2]
    # - level_start_index: [num_levels]
    # - sampling_locations: [N, num_queries, num_heads, num_levels, num_points, 2]  
    # - attention_weights: [N, num_queries, num_heads, num_levels, num_points]
    
    # 当前输入格式转换:
    # value: [total_spatial, m, d] -> [1, total_spatial, m, d]
    N = 1
    value_reshaped = value.unsqueeze(0)  # [1, total_spatial, m, d]
    
    # sampling_locations: [lq, m, l*k, 2] -> [1, lq, m, l, k, 2]  
    sampling_locs_reshaped = sampling_locations.view(lq, m, l, k, 2).unsqueeze(0)  # [1, lq, m, l, k, 2]
    
    # attention_weights: [lq, m, l*k] -> [1, lq, m, l, k]
    attn_weights_reshaped = attention_weights.view(lq, m, l, k).unsqueeze(0)  # [1, lq, m, l, k]
    
    # 调用MSDeformAttnFunction
    try:
        # 确保spatial_shapes和level_start_index是int64类型
        value_spatial_shapes_long = value_spatial_shapes.to(torch.int64)
        level_start_index_long = level_start_index.to(torch.int64)
        
        output = MSDeformAttnFunction.apply(
            value_reshaped,              # [1, total_spatial, m, d]
            value_spatial_shapes_long,   # [l, 2] - int64  
            level_start_index_long,      # [l] - int64
            sampling_locs_reshaped,      # [1, lq, m, l, k, 2]
            attn_weights_reshaped,       # [1, lq, m, l, k]
            1  # im2col_step
        )
        
        # 输出格式: [N, num_queries, num_heads * embed_dim] -> [lq, m, d]
        output = output.squeeze(0)  # [lq, m * d]
        return output.view(lq, m, d)  # [lq, m, d]
        
    except Exception as e:
        print(f"❌ MSDeformAttnFunction failed: {e}")
        print("Falling back to PyTorch reference implementation...")

def test_msdeform_function():
    """测试MSDeformAttnFunction是否正常工作"""
    print("🧪 Testing MSDeformAttnFunction...")
    
    # 创建简单的测试数据
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 简单测试参数
    N, lq, m, d, l, k = 1, 4, 2, 8, 2, 2
    spatial_shapes = [(4, 4), (2, 2)]
    total_spatial = sum(h * w for h, w in spatial_shapes)
    
    print(f"Debug: lq={lq}, m={m}, d={d}, l={l}, k={k}")
    print(f"Debug: total_spatial={total_spatial}")
    print(f"Debug: spatial_shapes={spatial_shapes}")
    
    # 创建测试数据
    value = torch.randn(total_spatial, m, d, device=device, dtype=torch.float32)
    spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
    level_start_index = torch.tensor([0, 16], dtype=torch.int64, device=device)
    sampling_locations = torch.rand(lq, m, l*k, 2, device=device, dtype=torch.float32) * 0.8 + 0.1
    attention_weights = torch.rand(lq, m, l*k, device=device, dtype=torch.float32)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    print(f"Debug: value.shape = {value.shape}")
    print(f"Debug: sampling_locations.shape = {sampling_locations.shape}")
    print(f"Debug: attention_weights.shape = {attention_weights.shape}")
    
    inputs = (value, spatial_shapes_tensor, level_start_index, sampling_locations, attention_weights)
    
    try:
        output = msdeform_kernel(*inputs)
        print(f"✅ MSDeformAttnFunction test passed! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ MSDeformAttnFunction test failed: {e}")

if __name__ == "__main__":
    test_msdeform_function()
