import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Layer Normalization 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同hidden size和sequence length
            self.test_shapes = [
                (1024, 512),      # 小批次小特征
                (2048, 768),      # GPT-2 small style
                (4096, 1024),     # GPT-2 medium style
                (1024, 2048),     # GPT-2 large style
                (512, 4096),      # GPT-3 style
                (256, 5120),      # LLaMA 7B style
                (128, 8192),      # LLaMA 30B style
                (64, 12288),      # LLaMA 65B style
                (32, 16384),      # 超大hidden size
                (8192, 256),      # 大批次小特征
                (4096, 3072),     # BERT large style
                (1536, 768),      # BERT base style
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_void_p,  # weight (GPU pointer)
        ctypes.c_void_p,  # bias (GPU pointer)
        ctypes.c_void_p,  # mean (GPU pointer)
        ctypes.c_void_p,  # rstd (GPU pointer)
        ctypes.c_int,     # stride
        ctypes.c_int,     # N (hidden size)
        ctypes.c_float,   # eps
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    M, N = shape
    eps = 1e-5

    # 创建输入张量
    x = torch.randn((M, N), dtype=torch.float16, device="cuda")
    weight = torch.randn((N,), dtype=torch.float16, device="cuda")
    bias = torch.zeros((N,), dtype=torch.float16, device="cuda")

    # 创建输出张量和中间结果
    y_cuda = torch.empty_like(x)
    mean_cuda = torch.empty((M,), dtype=torch.float32, device="cuda")
    rstd_cuda = torch.empty((M,), dtype=torch.float32, device="cuda")

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        x,
        y_cuda,
        weight,
        bias,
        mean_cuda,
        rstd_cuda,
        x.stride(0),  # stride
        N,            # hidden size
        eps           # epsilon
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [x, (N,), weight, bias, eps]

    # CUDA输出张量用于结果比较
    cuda_output_tensors = [y_cuda]

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

def torch_kernel(x: torch.Tensor, normalized_shape: Tuple[int, ...], weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch参考实现：layer normalization"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)