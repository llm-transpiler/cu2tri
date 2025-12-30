import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Liger-Kernel RMS Norm 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同transformer配置
            self.test_shapes = [
                (1, 512, 768),      # 小模型 (7B)
                (2, 256, 512),      # 中等批次
                (4, 128, 1024),     # 中等模型 (13B)
                (8, 64, 256),       # 大批次
                (16, 32, 128),      # 超大批次
                (1, 1024, 2048),    # 大模型 (70B)
                (2, 2048, 1536),    # 长序列大模型
                (4, 512, 4096),     # 超大hidden size
                (1, 4096, 1024),    # 超长序列
                (8, 256, 768),      # 中等配置
                (32, 16, 64),       # 小token大批次
                (1, 8192, 512),     # 超长序列
                (2, 128, 3072),     # 特殊配置
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # hidden_states (GPU pointer)
        ctypes.c_void_p,  # weight (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_float,   # eps
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # hidden_size
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    batch_size, seq_len, hidden_size = shape

    # 创建hidden_states张量
    hidden_states = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)

    # 创建weight张量
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)

    # 创建输出张量
    output_cuda = torch.empty_like(hidden_states)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        hidden_states,
        weight,
        output_cuda,
        1e-6,  # eps
        batch_size,
        seq_len,
        hidden_size
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [hidden_states, weight]

    # CUDA输出张量用于结果比较
    cuda_output_tensors = [output_cuda]

    return cuda_all_inputs, torch_all_inputs, cuda_output_tensors

def get_cuda_torch_inputs(params: Params):
    """为默认形状创建输入数据（保持向后兼容）"""
    return get_cuda_torch_inputs_for_shape(params.test_shapes[0])

def get_all_cuda_torch_inputs(params: Params):
    """为所有测试形状创建输入数据"""
    all_test_data = []

    for shape in params.test_shapes:
        test_data = get_cuda_torch_inputs_for_shape(shape)
        all_test_data.append((shape, test_data))

    return all_test_data

def cuda_output_tensor_transform(cuda_output):
    return cuda_output

def torch_kernel(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：RMS normalization"""
    # 计算RMS: sqrt(mean(x^2) + eps)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    rms = torch.rsqrt(variance + 1e-6)

    # 应用RMS norm
    return hidden_states * rms * weight

def torch_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """别名函数，为了兼容性"""
    return torch_kernel(hidden_states, weight)