import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """FlagGems Add 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同大小
            self.test_shapes = [
                (1, 15, 64),       # 小型
                (2, 15, 64),       # 批次变化
                (4, 15, 64),       # 中等批次
                (8, 15, 64),       # 大批次
                (1, 64, 128),      # 不同形状
                (2, 256, 512),     # 中等大小
                (4, 1024, 1024),   # 大型
                (1, 32, 32),       # 小形状
                (1, 128, 256),     # 中等形状
                (16, 64, 256),     # 大批次大形状
                (1, 512, 768),     # 768 hidden size
                (2, 1024, 1024),   # 1K hidden size
                (1, 2048, 1536),   # 1.5K hidden size
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # X (GPU pointer)
        ctypes.c_void_p,  # Y (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int      # N (total elements)
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    N = torch.prod(torch.tensor(shape)).item()

    # 创建给定形状的输入张量
    x = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)
    y = torch.randn(shape, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5)

    # 创建输出张量
    output_cuda = torch.empty_like(x)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        x,
        y,
        output_cuda,
        N
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [x, y]

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