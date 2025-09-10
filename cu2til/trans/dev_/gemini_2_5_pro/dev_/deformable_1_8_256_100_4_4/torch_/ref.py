import torch
import torch.nn.functional as F

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    官方PyTorch参考实现 (从MultiScaleDeformableAttention包中提取)
    
    Args:
        value: (N_, S_, M_, D_) - batch, spatial_size, num_heads, embed_dim
        value_spatial_shapes: spatial shapes tensor/list
        sampling_locations: (N_, Lq_, M_, L_, P_, 2) - batch, num_queries, num_heads, num_levels, num_points, 2
        attention_weights: (N_, Lq_, M_, L_, P_) - batch, num_queries, num_heads, num_levels, num_points
        
    Returns:
        output: (N_, Lq_, M_*D_) - batch, num_queries, num_heads*embed_dim
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


def torch_kernel(*args):
    """
    适配器函数：将我们的输入格式转换为官方PyTorch实现的格式
    
    Args:
        value: (S_, M_, D_) - spatial_size, num_heads, embed_dim  
        value_spatial_shapes: (L_, 2) - num_levels x [height, width]
        level_start_index: (L_,) - level start indices (unused in this implementation)
        sampling_locations: (Lq_, M_, L_*P_, 2) - num_queries, num_heads, num_levels*num_points, 2
        attention_weights: (Lq_, M_, L_*P_) - num_queries, num_heads, num_levels*num_points
    
    Returns:
        output: (Lq_, M_, D_) - num_queries, num_heads, embed_dim
    """
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = args
    
    # 从输入张量推断参数
    S_, M_, D_ = value.shape  # spatial_size, num_heads, embed_dim
    Lq_, M_check, LxP_, _ = sampling_locations.shape  # num_queries, num_heads, num_levels*num_points, 2
    L_ = len(level_start_index)  # num_levels
    P_ = LxP_ // L_  # num_points per level
    
    assert M_ == M_check, f"Head数不匹配: value={M_}, sampling_locations={M_check}"
    assert LxP_ == L_ * P_, f"采样点数不匹配: {LxP_} != {L_} * {P_}"
    
    # 转换为官方实现的输入格式
    
    # 1. 为value添加batch维度: (S_, M_, D_) -> (1, S_, M_, D_)
    value_batched = value.unsqueeze(0)
    
    # 2. 重塑sampling_locations: (Lq_, M_, L_*P_, 2) -> (1, Lq_, M_, L_, P_, 2)
    sampling_locations_reshaped = sampling_locations.view(Lq_, M_, L_, P_, 2).unsqueeze(0)
    
    # 3. 重塑attention_weights: (Lq_, M_, L_*P_) -> (1, Lq_, M_, L_, P_)
    attention_weights_reshaped = attention_weights.view(Lq_, M_, L_, P_).unsqueeze(0)
    
    # 4. 调用官方PyTorch实现
    output_batched = ms_deform_attn_core_pytorch(
        value_batched, 
        value_spatial_shapes, 
        sampling_locations_reshaped, 
        attention_weights_reshaped
    )
    
    # 5. 移除batch维度并重塑输出: (1, Lq_, M_*D_) -> (Lq_, M_, D_)
    output = output_batched.squeeze(0)  # (Lq_, M_*D_)
    output = output.view(Lq_, M_, D_)   # (Lq_, M_, D_)
    
    return output


def test_torch_kernel():
    """测试torch_kernel函数"""
    print("🧪 测试官方PyTorch参考实现...")
    
    # 设置测试参数
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试参数
    lq, m, d, l, k = 10, 4, 8, 2, 2
    spatial_shapes = [(8, 8), (4, 4)]
    total_spatial = sum(h * w for h, w in spatial_shapes)
    
    # 计算level start index
    level_starts = []
    current_start = 0
    for h, w in spatial_shapes:
        level_starts.append(current_start)
        current_start += h * w
    
    try:
        # 创建测试数据
        value = torch.randn(total_spatial, m, d, dtype=torch.float32, device=device) * 0.01
        value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int64, device=device)
        level_start_index = torch.tensor(level_starts, dtype=torch.int64, device=device)
        sampling_locations = torch.rand(lq, m, l*k, 2, dtype=torch.float32, device=device)
        sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)  # 确保在[0,1]范围内
        attention_weights = torch.rand(lq, m, l*k, dtype=torch.float32, device=device) + 1e-5
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        print(f"  - 输入参数: lq={lq}, m={m}, d={d}, l={l}, k={k}")
        print(f"  - 空间尺寸: {spatial_shapes}")
        print(f"  - 总空间大小: {total_spatial}")
        
        # 测试函数
        inputs = (value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights)
        output = torch_kernel(*inputs)
        
        print(f"  ✅ 成功! 输出形状: {output.shape}")
        print(f"  - 输出统计: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
        print(f"  - 输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
        # 检查输出的合理性
        assert output.shape == (lq, m, d), f"输出形状错误: 期望 {(lq, m, d)}, 得到 {output.shape}"
        assert torch.isfinite(output).all(), "输出包含NaN或Inf"
        
        print("  ✅ 所有检查通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 官方PyTorch参考实现测试")
    print("=" * 50)
    success = test_torch_kernel()
    if success:
        print("\n🎉 官方PyTorch参考实现工作正常！")
    else:
        print("\n❌ 测试失败，请检查实现")
