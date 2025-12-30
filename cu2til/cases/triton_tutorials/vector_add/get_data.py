import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Vector Add 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，使用更小的配置避免内存问题
            self.test_shapes = [
                (1024,),        # 小向量
                (4096,),        # 中小向量
                (16384,),       # 中等向量
                (65536,),       # 大向量
                (262144,),      # 超大向量
                (98432,),       # tutorial中的示例大小
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # x (GPU pointer)
        ctypes.c_void_p,  # y (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int      # n_elements
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    n_elements = shape[0] if len(shape) == 1 else torch.prod(torch.tensor(shape)).item()

    # 创建输入向量
    x = torch.randn(shape, dtype=torch.float32, device="cuda")
    y = torch.randn(shape, dtype=torch.float32, device="cuda")

    # 创建输出向量
    output_cuda = torch.empty_like(x)

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        x,
        y,
        output_cuda,
        n_elements
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