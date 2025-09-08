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
    适配器函数：处理batched输入数据并转换为官方PyTorch实现的格式
    
    Args:
        value: (N_, S_, M_, D_) - batch, spatial_size, num_heads, embed_dim (已经是batched)
        value_spatial_shapes: (L_, 2) - num_levels x [height, width]
        level_start_index: (L_,) - level start indices (unused in this implementation)
        sampling_locations: (N_, Lq_, M_, L_*P_, 2) - batch, num_queries, num_heads, num_levels*num_points, 2
        attention_weights: (N_, Lq_, M_, L_*P_) - batch, num_queries, num_heads, num_levels*num_points
    
    Returns:
        output: (N_, Lq_, M_, D_) - batch, num_queries, num_heads, embed_dim (batched)
    """
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = args
    
    # 处理混合格式输入：value可能是展平的，其他tensor是batched的
    if len(value.shape) == 4:
        # 标准batched格式: [N, S, M, D]
        N_, S_, M_, D_ = value.shape
        # print(f"   Processing standard batched input: N={N_}, S={S_}, M={M_}, D={D_}")
    elif len(value.shape) == 3:
        # value可能是展平的 [N*S, M, D] 或非batched [S, M, D]
        # 通过checking sampling_locations来推断batch size
        if len(sampling_locations.shape) == 5:
            N_check, Lq_, M_check, LxP_, _ = sampling_locations.shape
            # 这是展平的batched格式
            NS_, M_, D_ = value.shape
            S_ = NS_ // N_check
            N_ = N_check
            # 重塑value为标准batched格式
            value = value.view(N_, S_, M_, D_)
            # print(f"   Reconstructed batched input from flattened: N={N_}, S={S_}, M={M_}, D={D_}")
        else:
            # 真正的非batched格式
            S_, M_, D_ = value.shape
            N_ = 1
            value = value.unsqueeze(0)
            sampling_locations = sampling_locations.unsqueeze(0)
            attention_weights = attention_weights.unsqueeze(0)
            # print(f"   Added batch dimension: N={N_}, S={S_}, M={M_}, D={D_}")
    else:
        raise ValueError(f"Unexpected value tensor shape: {value.shape}")
    
    # 从输入张量推断参数
    if len(sampling_locations.shape) == 5:
        N_check, Lq_, M_check, LxP_, _ = sampling_locations.shape
    else:
        raise ValueError(f"Unexpected sampling_locations shape: {sampling_locations.shape}")
        
    L_ = len(level_start_index)  # num_levels
    P_ = LxP_ // L_  # num_points per level
    
    assert N_ == N_check, f"Batch数不匹配: value={N_}, sampling_locations={N_check}"
    assert M_ == M_check, f"Head数不匹配: value={M_}, sampling_locations={M_check}"
    assert LxP_ == L_ * P_, f"采样点数不匹配: {LxP_} != {L_} * {P_}"
    
    # print(f"   Batch processing: N={N_}, Lq={Lq_}, M={M_}, D={D_}, L={L_}, P={P_}")
    
    # 重塑sampling_locations和attention_weights以匹配官方实现
    # (N_, Lq_, M_, L_*P_, 2) -> (N_, Lq_, M_, L_, P_, 2)
    sampling_locations_reshaped = sampling_locations.view(N_, Lq_, M_, L_, P_, 2)
    
    # (N_, Lq_, M_, L_*P_) -> (N_, Lq_, M_, L_, P_)
    attention_weights_reshaped = attention_weights.view(N_, Lq_, M_, L_, P_)
    
    # 调用官方PyTorch实现 (已经是正确的batched格式)
    output_batched = ms_deform_attn_core_pytorch(
        value, 
        value_spatial_shapes, 
        sampling_locations_reshaped, 
        attention_weights_reshaped
    )
    
    # 重塑输出: (N_, Lq_, M_*D_) -> (N_, Lq_, M_, D_)
    output = output_batched.view(N_, Lq_, M_, D_)
    
    # print(f"   Output shape: {output.shape}")
    return output