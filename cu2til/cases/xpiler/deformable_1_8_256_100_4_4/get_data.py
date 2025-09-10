import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Deformable Attention operation parameters"""
    num_queries: int = 100  # lq
    num_heads: int = 8      # m  
    embed_dim: int = 256    # d
    num_levels: int = 4     # l
    num_points: int = 4     # k

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # value (GPU pointer)
        ctypes.c_void_p,  # value_spatial_shapes (GPU pointer)
        ctypes.c_void_p,  # level_start_index (GPU pointer)
        ctypes.c_void_p,  # sampling_locations (GPU pointer)
        ctypes.c_void_p,  # attention_weights (GPU pointer)
        ctypes.c_void_p   # output (GPU pointer)
    ]

def get_cuda_torch_inputs(params: Params):
    """Create input data for both CUDA and PyTorch implementations"""
    torch.manual_seed(SEED)
    
    # 定义4个level的空间形状 (height, width)
    spatial_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int32, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int32, device="cuda")
    
    # 创建value tensor: [total_spatial_size, m, d]
    value = torch.randn(total_spatial, params.num_heads, params.embed_dim, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # 创建采样位置: [lq, m, l*k, 2] (归一化坐标 0-1)
    sampling_locations = torch.rand(params.num_queries, params.num_heads, params.num_levels*params.num_points, 2, dtype=torch.float32, device="cuda")
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 创建注意力权重: [lq, m, l*k]
    attention_weights = torch.randn(params.num_queries, params.num_heads, params.num_levels*params.num_points, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # 创建输出张量: [lq, m, d]
    output_tensor = torch.empty(params.num_queries, params.num_heads, params.embed_dim, dtype=torch.float32, device="cuda")
    
    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        value,
        value_spatial_shapes, 
        level_start_index,
        sampling_locations,
        attention_weights,
        output_tensor
    ]
    
    # PyTorch输入：5个张量参数
    torch_all_inputs = [value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights]
    
    # CUDA输出张量用于结果比较
    cuda_output_tensors = [output_tensor]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_output_tensor_transform(cuda_output):
    return cuda_output
