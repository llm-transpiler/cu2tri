import torch
import ctypes
from dataclasses import dataclass
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Deformable Attention operation parameters (batched version)"""
    batch_size: int = 4     # n (kernel硬编码的batch数量)
    num_queries: int = 200  # lq
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
    """Create batched input data for both CUDA and PyTorch implementations"""
    torch.manual_seed(SEED)
    
    n = params.batch_size  # 4
    lq = params.num_queries  # 200
    m = params.num_heads    # 8
    d = params.embed_dim    # 256
    l = params.num_levels   # 4
    k = params.num_points   # 4
    
    # 从kernel内存访问分析得到的正确spatial size (for d=256)
    kernel_value_stride = 26830848
    correct_spatial_size = kernel_value_stride // (m * d)  # 13101
    
    # 定义精确匹配13101的空间形状配置
    base_size = 114  # 114*114 = 12996
    remaining = correct_spatial_size - base_size * base_size  # 105
    spatial_shapes = [(base_size, base_size), (remaining-2, 1), (1, 1), (1, 1)]  # 12996 + 103 + 1 + 1 = 13101
    
    value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.int32, device="cuda")
    
    # 计算每个level的起始索引
    level_starts = []
    total_spatial = 0
    for h, w in spatial_shapes:
        level_starts.append(total_spatial)
        total_spatial += h * w
    level_start_index = torch.tensor(level_starts, dtype=torch.int32, device="cuda")
    
    # 使用kernel期望的确切大小
    actual_spatial_size = correct_spatial_size
    
    # 创建正确的批处理数据
    # 注意：这里使用的是批处理版本的张量形状，与单样本版本不同
    value = torch.randn(n, actual_spatial_size, m, d, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    
    # 批处理版本的采样位置: [n, lq, m, l*k, 2] 
    sampling_locations = torch.rand(n, lq, m, l*k, 2, dtype=torch.float32, device="cuda")
    sampling_locations = torch.clamp(sampling_locations, 0.1, 0.9)
    
    # 批处理版本的注意力权重: [n, lq, m, l*k]
    attention_weights = torch.randn(n, lq, m, l*k, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    attention_weights = torch.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    # 批处理版本的输出张量: [n, lq, m, d]
    output_tensor = torch.empty(n, lq, m, d, dtype=torch.float32, device="cuda")
    
    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        value,
        value_spatial_shapes, 
        level_start_index,
        sampling_locations,
        attention_weights,
        output_tensor
    ]
    
    # PyTorch输入：批处理版本的5个张量参数
    torch_all_inputs = [value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights]
    
    # CUDA输出张量用于结果比较
    cuda_output_tensors = [output_tensor]
    
    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def cuda_output_tensor_transform(cuda_output):
    return cuda_output