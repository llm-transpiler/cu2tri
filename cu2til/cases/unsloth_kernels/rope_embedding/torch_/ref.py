import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Unsloth RoPE Embedding 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同transformer配置
            self.test_shapes = [
                (1, 512, 32, 64),    # 小模型
                (2, 1024, 16, 128),  # 中等批次
                (4, 256, 8, 256),    # 深度模型
                (1, 2048, 32, 128),  # 长序列
                (8, 128, 16, 64),    # 大批次
                (16, 64, 8, 128),    # 超大批次
                (1, 4096, 32, 64),   # 超长序列
                (2, 512, 24, 96),    # 7B模型配置
                (1, 1024, 16, 256),  # 宽embedding
                (4, 2048, 32, 256),  # 大模型长序列
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # Q (GPU pointer)
        ctypes.c_void_p,  # cos (GPU pointer)
        ctypes.c_void_p,  # sin (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # n_heads
        ctypes.c_int,     # head_dim
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    batch_size, seq_len, n_heads, head_dim = shape

    # 创建Q张量
    Q = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)

    # 创建cos和sin（用于RoPE）
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    freqs = torch.einsum('i,j->ij', positions, inv_freq)
    cos = torch.cos(freqs)  # [seq_len, head_dim//2]
    sin = torch.sin(freqs)  # [seq_len, head_dim//2]

    # 创建输出张量
    output_cuda = torch.empty_like(Q)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        Q,
        cos,
        sin,
        output_cuda,
        batch_size,
        seq_len,
        n_heads,
        head_dim
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [Q, cos, sin]

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

def torch_kernel(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：RoPE embedding"""
    batch_size, seq_len, n_heads, head_dim = Q.shape

    # 分割Q用于rotate_half
    Q1, Q2 = Q[..., :head_dim//2], Q[..., head_dim//2:]
    Q_rotated = torch.cat([-Q2, Q1], dim=-1)

    # 扩展cos和sin以匹配Q的维度
    # cos和sin的形状为[seq_len, head_dim//2]，需要扩展到[batch_size, seq_len, n_heads, head_dim//2]
    cos_expanded = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)
    sin_expanded = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, n_heads, -1)

    # 拼接cos和sin以获得完整的head维度
    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)

    # 应用RoPE公式
    Q_rope = Q * cos_full + Q_rotated * sin_full

    return Q_rope