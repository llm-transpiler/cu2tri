import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Matrix Multiplication 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同矩阵尺寸
            self.test_shapes = [
                (128, 128, 128),    # 小方阵
                (256, 256, 256),    # 中等方阵
                (512, 512, 512),    # 大方阵
                (1024, 1024, 1024), # 超大方阵
                (64, 512, 256),     # 小批次大矩阵
                (512, 64, 128),     # 大批次小矩阵
                (2048, 1024, 512),  # 长方形矩阵
                (768, 2304, 768),   # LLaMA style
                (1024, 768, 3072),  # GPT style
                (4096, 4096, 1024), # 超大矩阵
                (32, 8192, 512),    # 单行超长序列
                (8192, 32, 64),     # 超多行小特征
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # Matrix A (GPU pointer)
        ctypes.c_void_p,  # Matrix B (GPU pointer)
        ctypes.c_void_p,  # Matrix C (GPU pointer)
        ctypes.c_int,     # M (rows of A)
        ctypes.c_int,     # N (cols of B)
        ctypes.c_int,     # K (cols of A / rows of B)
        ctypes.c_int,     # stride_am
        ctypes.c_int,     # stride_ak
        ctypes.c_int,     # stride_bk
        ctypes.c_int,     # stride_bn
        ctypes.c_int,     # stride_cm
        ctypes.c_int,     # stride_cn
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    M, N, K = shape

    # 创建输入矩阵 A (M, K) 和 B (K, N)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b = torch.randn((K, N), dtype=torch.float16, device="cuda")

    # 创建输出矩阵 C (M, N)
    c_cuda = torch.empty((M, N), dtype=torch.float16, device="cuda")

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        a,
        b,
        c_cuda,
        M, N, K,
        a.stride(0), a.stride(1),  # stride_am, stride_ak
        b.stride(0), b.stride(1),  # stride_bk, stride_bn
        c_cuda.stride(0), c_cuda.stride(1)  # stride_cm, stride_cn
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [a, b]

    # CUDA输出张量用于结果比较
    cuda_output_tensors = [c_cuda]

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

def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：matrix multiplication"""
    return torch.matmul(a, b)