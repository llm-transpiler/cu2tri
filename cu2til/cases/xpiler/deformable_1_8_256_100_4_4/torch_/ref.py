import torch
import torch.nn.functional as F

def torch_kernel(*args):
    """
    基于 ms_deform_attn_func.py 逻辑重新实现，匹配 kernel.cu 的具体行为
    修正了坐标系统和双线性插值逻辑
    """
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = args
    
    # 从CUDA kernel分析得到的参数
    lq = 100  # num_queries  
    m = 8     # num_heads
    d = 256   # embed_dim
    l = 4     # num_levels
    k = 4     # num_points per level
    
    # 重塑输入张量以匹配CUDA kernel的数据布局
    # sampling_locations: [lq, m, l*k, 2] -> [lq, m, l, k, 2]
    sampling_locs = sampling_locations.view(lq, m, l, k, 2)
    # attention_weights: [lq, m, l*k] -> [lq, m, l, k] 
    attn_weights = attention_weights.view(lq, m, l, k)
    
    # 输出张量: [lq, m, d]
    output = torch.zeros(lq, m, d, device=value.device, dtype=value.dtype)
    
    # 遍历每个query和head (对应CUDA kernel中的 blockIdx.x 和 blockIdx.z)
    for query_idx in range(lq):
        for head_idx in range(m):
            attention_sum = torch.zeros(d, device=value.device, dtype=value.dtype)
            
            # 遍历每个level (对应CUDA kernel中的 i 循环)
            for level_idx in range(l):
                # 获取当前level的空间尺寸
                height = value_spatial_shapes[level_idx, 0].item()
                width = value_spatial_shapes[level_idx, 1].item() 
                level_start = level_start_index[level_idx].item()
                
                # 遍历每个采样点 (对应CUDA kernel中的 k 循环)
                for point_idx in range(k):
                    # 获取采样位置 (归一化坐标 0-1)
                    # 注意：CUDA kernel中 xy[1] 对应 Y, xy[0] 对应 X
                    xy = sampling_locs[query_idx, head_idx, level_idx, point_idx]  # [2]
                    y_norm, x_norm = xy[0].item(), xy[1].item()  # 注意顺序！
                    
                    # 转换为像素坐标 (匹配CUDA kernel逻辑)
                    # xy_grid[0] = ((xy[0] * ((float)height_width[0])) - 5.000000e-01f);
                    # xy_grid[1] = ((xy[1] * ((float)height_width[1])) - 5.000000e-01f);
                    y_pix = y_norm * height - 0.5  # 对应xy_grid[0]
                    x_pix = x_norm * width - 0.5   # 对应xy_grid[1]
                    
                    # 计算四个邻近像素坐标
                    # xy_rounded[0] = ((int)floorf(xy_grid[0]));
                    # xy_rounded[1] = (xy_rounded[0] + 1);
                    # xy_rounded[2] = ((int)floorf(xy_grid[1]));
                    # xy_rounded[3] = (xy_rounded[2] + 1);
                    y0 = int(torch.floor(torch.tensor(y_pix)).item())  # xy_rounded[0]
                    y1 = y0 + 1                                        # xy_rounded[1]
                    x0 = int(torch.floor(torch.tensor(x_pix)).item())  # xy_rounded[2]
                    x1 = x0 + 1                                        # xy_rounded[3]
                    
                    # 初始化四个角的值
                    corner_values = torch.zeros(4, d, device=value.device, dtype=value.dtype)
                    
                    # 四个角的坐标和对应的corner_values索引
                    # 按照CUDA kernel的检查顺序
                    corners = [
                        (y0, x0, 0),  # corner_values[ii_d_2] - 对应 (xy_rounded[0], xy_rounded[2])
                        (y0, x1, 1),  # corner_values[ii_d_3 + 4] - 对应 (xy_rounded[0], xy_rounded[3])
                        (y1, x0, 2),  # corner_values[ii_d_5 + 8] - 对应 (xy_rounded[1], xy_rounded[2])
                        (y1, x1, 3),  # corner_values[ii_d_7 + 12] - 对应 (xy_rounded[1], xy_rounded[3])
                    ]
                    
                    for y, x, idx in corners:
                        # 边界检查 (匹配CUDA kernel的边界条件)
                        if 0 <= y < height and 0 <= x < width:
                            # 计算value tensor中的索引
                            # value_[((((((value_level_start_index_[i] * 2048) +
                            #           ((xy_rounded[0] * height_width[1]) * 2048)) +
                            #          (xy_rounded[2] * 2048)) +
                            #         (((int)blockIdx.z) * 256)) +
                            #        (((int)threadIdx.x) * 4)) +
                            #       ii_d_2)]
                            spatial_idx = level_start + y * width + x
                            corner_values[idx] = value[spatial_idx, head_idx, :]
                    
                    # 双线性插值权重计算 (匹配CUDA kernel的权重计算)
                    # (((float)xy_rounded[1]) - xy_grid[0]) 对应 (y1 - y_pix)
                    # (xy_grid[0] - ((float)xy_rounded[0])) 对应 (y_pix - y0)  
                    # (((float)xy_rounded[3]) - xy_grid[1]) 对应 (x1 - x_pix)
                    # (xy_grid[1] - ((float)xy_rounded[2])) 对应 (x_pix - x0)
                    w_y1 = y1 - y_pix  # 1.0 - (y_pix - y0)
                    w_y0 = y_pix - y0  # (y_pix - y0)
                    w_x1 = x1 - x_pix  # 1.0 - (x_pix - x0) 
                    w_x0 = x_pix - x0  # (x_pix - x0)
                    
                    # 双线性插值 (按照CUDA kernel的公式)
                    # ((((corner_values[ii_d_9] * (((float)xy_rounded[1]) - xy_grid[0])) * (((float)xy_rounded[3]) - xy_grid[1])) +
                    #   ((corner_values[(ii_d_9 + 8)] * (xy_grid[0] - ((float)xy_rounded[0]))) * (((float)xy_rounded[3]) - xy_grid[1]))) +
                    #   ((corner_values[(ii_d_9 + 4)] * (((float)xy_rounded[1]) - xy_grid[0])) * (xy_grid[1] - ((float)xy_rounded[2])))) +
                    #   ((corner_values[(ii_d_9 + 12)] * (xy_grid[0] - ((float)xy_rounded[0]))) * (xy_grid[1] - ((float)xy_rounded[2]))))
                    
                    interpolated = (corner_values[0] * w_y1 * w_x1 +  # (y0, x0)
                                  corner_values[1] * w_y1 * w_x0 +  # (y0, x1)  
                                  corner_values[2] * w_y0 * w_x1 +  # (y1, x0)
                                  corner_values[3] * w_y0 * w_x0)   # (y1, x1)
                    
                    # 用注意力权重加权累积
                    weight = attn_weights[query_idx, head_idx, level_idx, point_idx]
                    attention_sum += interpolated * weight
            
            # 存储结果
            output[query_idx, head_idx, :] = attention_sum
    
    return output


def torch_kernel_vectorized(*args):
    """
    向量化版本 - 基于 ms_deform_attn_core_pytorch 的逻辑但匹配 kernel.cu 的参数
    这个版本应该更高效，但逻辑更复杂
    """
    value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights = args
    
    # 参数
    N_, Lq_, M_, L_, P_ = 1, 100, 8, 4, 4
    D_ = 256
    
    # 重新组织数据以匹配原始 ms_deform_attn_core_pytorch 的输入格式
    # value: [total_spatial, M, D] -> [N, S, M, D]
    S_ = value.shape[0]  # total spatial size
    value_reshaped = value.unsqueeze(0)  # [1, S, M, D]
    
    # sampling_locations: [Lq, M, L*P, 2] -> [N, Lq, M, L, P, 2]
    sampling_locs = sampling_locations.view(Lq_, M_, L_, P_, 2).unsqueeze(0)  # [1, Lq, M, L, P, 2]
    
    # attention_weights: [Lq, M, L*P] -> [N, Lq, M, L, P]  
    attn_weights = attention_weights.view(Lq_, M_, L_, P_).unsqueeze(0)  # [1, Lq, M, L, P]
    
    # 将value按照spatial shapes分割
    value_list = []
    for i, (H_, W_) in enumerate(value_spatial_shapes):
        start_idx = level_start_index[i].item()
        end_idx = start_idx + H_.item() * W_.item()
        level_value = value_reshaped[:, start_idx:end_idx, :, :]  # [1, H*W, M, D]
        value_list.append(level_value)
    
    # 计算采样网格 (归一化坐标转换为grid_sample坐标)
    sampling_grids = 2 * sampling_locs - 1  # [1, Lq, M, L, P, 2]
    
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = H_.item(), W_.item()
        # [1, H*W, M, D] -> [1*M, D, H, W]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # [1, Lq, M, P, 2] -> [1*M, Lq, P, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # 双线性插值
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # 组合结果
    # [1, Lq, M, L, P] -> [1*M, 1, Lq, L*P]
    attention_weights_reshaped = attn_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # 加权求和
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights_reshaped).sum(-1).view(N_, M_*D_, Lq_)
    # [1, M*D, Lq] -> [Lq, M, D]
    return output.transpose(1, 2).contiguous().squeeze(0).view(Lq_, M_, D_)
