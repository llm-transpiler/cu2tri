import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Fused Softmax 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同序列长度
            self.test_shapes = [
                (128, 1024),      # 小序列
                (256, 512),       # 中等序列
                (512, 2048),      # 大序列
                (1024, 1024),     # 大批次中等序列
                (64, 4096),       # 小批次大序列
                (2048, 256),      # 超大批次小序列
                (32, 8192),       # 超长序列
                (1536, 1536),     # 正方形大矩阵
                (4096, 512),      # 非常大的批次
                (1, 32768),       # 超长单序列
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # rows
        ctypes.c_int,     # cols
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    rows, cols = shape

    # 创建输入矩阵
    x = torch.randn(shape, dtype=torch.float32, device="cuda")

    # 创建输出矩阵
    output_cuda = torch.empty_like(x)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        x,
        output_cuda,
        rows,
        cols
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [x]

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

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现：fused softmax"""
    return torch.softmax(x, dim=-1)