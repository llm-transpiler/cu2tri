import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Fused Attention 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，格式为 (batch_size, n_heads, seq_len, head_dim)
            # 使用更小、更安全的配置以避免内存问题
            self.test_shapes = [
                (1, 4, 256, 64),     # 小模型
                (2, 8, 512, 64),     # 中等模型
                (1, 12, 1024, 64),   # 大模型
                (1, 8, 512, 128),    # 大head维度
                (2, 6, 768, 64),     # BERT base风格
                (1, 4, 256, 96),     # 其他head维度
                (2, 2, 128, 128),    # 极小配置
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # Q (GPU pointer)
        ctypes.c_void_p,  # K (GPU pointer)
        ctypes.c_void_p,  # V (GPU pointer)
        ctypes.c_void_p,  # Output (GPU pointer)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # n_heads
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # head_dim
        ctypes.c_float,   # scale factor
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    batch_size, n_heads, seq_len, head_dim = shape
    scale = head_dim ** -0.5

    # 创建Q, K, V张量
    q = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")

    # 创建输出张量
    output_cuda = torch.empty_like(q)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        q,
        k,
        v,
        output_cuda,
        batch_size,
        n_heads,
        seq_len,
        head_dim,
        scale
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [q, k, v, scale]

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

def torch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch参考实现：scaled dot-product attention"""
    # 使用torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
    try:
        import torch.nn.functional as F
        # Reshape for scaled_dot_product_attention: (B, H, S, D) -> (B, S, H, D)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # 使用PyTorch的优化attention实现
        attn_output = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=0.0,
            is_causal=False,
            scale=scale
        )
        # Reshape back: (B, S, H, D) -> (B, H, S, D)
        return attn_output.transpose(1, 2)
    except ImportError:
        # Manual implementation for older PyTorch versions
        batch_size, n_heads, seq_len, head_dim = q.shape

        # Reshape for matrix multiplication: (B, H, S, D) -> (B*H, S, D)
        q_flat = q.view(batch_size * n_heads, seq_len, head_dim)
        k_flat = k.view(batch_size * n_heads, seq_len, head_dim)
        v_flat = v.view(batch_size * n_heads, seq_len, head_dim)

        # Compute attention scores: (B*H, S, S)
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values: (B*H, S, D)
        attn_output = torch.matmul(attn_weights, v_flat)

        # Reshape back: (B*H, S, D) -> (B, H, S, D)
        return attn_output.view(batch_size, n_heads, seq_len, head_dim)