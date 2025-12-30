import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Low Memory Dropout 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，涵盖不同张量大小
            # dropout通常用于全连接层，所以测试1D和2D张量
            self.test_shapes = [
                (1024,),          # 小向量
                (4096,),          # 中等向量
                (16384,),         # 大向量
                (65536,),         # 超大向量
                (256, 256),       # 小矩阵
                (512, 512),       # 中等矩阵
                (1024, 1024),     # 大矩阵
                (128, 2048),      # 长方形矩阵
                (2048, 128),      # 高矩阵
                (768, 768),       # 标准hidden size
                (1536, 1536),     # 大hidden size
                (256, 512, 512),  # 3D张量
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # input (GPU pointer)
        ctypes.c_void_p,  # output (GPU pointer)
        ctypes.c_int,     # n_elements
        ctypes.c_float,   # p (dropout probability)
        ctypes.c_int,     # seed
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    # Dropout概率
    p = 0.5  # 标准dropout概率

    # 创建输入张量
    if len(shape) == 1:
        x = torch.randn(shape, dtype=torch.float32, device="cuda")
    elif len(shape) == 2:
        x = torch.randn(shape, dtype=torch.float32, device="cuda")
    else:  # 3D
        x = torch.randn(shape, dtype=torch.float32, device="cuda")

    # 创建输出张量
    output_cuda = torch.empty_like(x)

    # 计算元素数量
    n_elements = x.numel()

    # 生成种子
    seed = 12345

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        x,
        output_cuda,
        n_elements,
        p,
        seed
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [x, p]

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

def torch_kernel(x: torch.Tensor, p: float) -> torch.Tensor:
    """PyTorch参考实现：dropout"""
    # PyTorch的dropout实现
    return torch.nn.functional.dropout(x, p=p, training=True)