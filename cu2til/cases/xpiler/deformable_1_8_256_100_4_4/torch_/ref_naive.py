import torch
import torch.nn.functional as F

def torch_kernel(*args):
    """PyTorch参考实现 for deformable attention - 匹配CUDA kernel实现"""
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
    
    # 输出张量
    output = torch.zeros(lq, m, d, device=value.device, dtype=value.dtype)
    
    # 遍历每个query和head
    for query_idx in range(lq):
        for head_idx in range(m):
            attention_sum = torch.zeros(d, device=value.device, dtype=value.dtype)
            
            # 遍历每个level
            for level_idx in range(l):
                # 获取当前level的空间尺寸
                height = value_spatial_shapes[level_idx, 0].item()
                width = value_spatial_shapes[level_idx, 1].item() 
                level_start = level_start_index[level_idx].item()
                
                # 遍历每个采样点
                for point_idx in range(k):
                    # 获取采样位置 (归一化坐标)
                    xy = sampling_locs[query_idx, head_idx, level_idx, point_idx]  # [2]
                    x_norm, y_norm = xy[1].item(), xy[0].item()  # 注意CUDA中的坐标顺序
                    
                    # 转换为像素坐标
                    x_pix = x_norm * height - 0.5
                    y_pix = y_norm * width - 0.5
                    
                    # 计算四个邻近像素坐标
                    x0 = int(torch.floor(torch.tensor(x_pix)).item())
                    x1 = x0 + 1
                    y0 = int(torch.floor(torch.tensor(y_pix)).item())
                    y1 = y0 + 1
                    
                    # 双线性插值权重
                    wx1 = x_pix - x0
                    wx0 = 1.0 - wx1  
                    wy1 = y_pix - y0
                    wy0 = 1.0 - wy1
                    
                    # 初始化四个角的值
                    corner_values = torch.zeros(4, d, device=value.device, dtype=value.dtype)
                    
                    # 检查边界并获取四个角的值
                    corners = [(x0, y0, 0), (x0, y1, 1), (x1, y0, 2), (x1, y1, 3)]
                    
                    for x, y, idx in corners:
                        if 0 <= x < height and 0 <= y < width:
                            # 计算value tensor中的索引
                            spatial_idx = level_start + x * width + y
                            corner_values[idx] = value[spatial_idx, head_idx, :]
                    
                    # 双线性插值
                    interpolated = (corner_values[0] * wx0 * wy0 +  # (x0, y0)
                                  corner_values[1] * wx0 * wy1 +  # (x0, y1)  
                                  corner_values[2] * wx1 * wy0 +  # (x1, y0)
                                  corner_values[3] * wx1 * wy1)   # (x1, y1)
                    
                    # 用注意力权重加权累积
                    weight = attn_weights[query_idx, head_idx, level_idx, point_idx]
                    attention_sum += interpolated * weight
            
            # 存储结果
            output[query_idx, head_idx, :] = attention_sum
    
    return output
