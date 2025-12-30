import torch
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from cu2til.tools.builder import SEED

@dataclass
class Params:
    """Triton Grouped GEMM 参数配置"""
    test_shapes: List[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.test_shapes is None:
            # 默认测试形状，格式为 (group_size, M, N, K)
            # 选择更保守的配置以避免内存问题
            self.test_shapes = [
                (2, 512, 512, 512),    # 2个小矩阵
                (4, 256, 256, 256),    # 4个小矩阵
                (3, 1024, 768, 768),   # GPT风格的矩阵组
                (5, 768, 2304, 768),   # LLaMA风格的矩阵组
                (2, 2048, 1024, 1024), # 2个中等矩阵
                (6, 512, 2048, 512),   # 不同尺寸的矩阵组
                (4, 384, 1536, 384),   # BERT base风格
                (3, 3072, 3072, 3072), # 大方阵组
            ]

def get_cuda_argtypes():
    return [
        ctypes.c_void_p,  # group_a_ptrs (GPU pointer)
        ctypes.c_void_p,  # group_b_ptrs (GPU pointer)
        ctypes.c_void_p,  # group_c_ptrs (GPU pointer)
        ctypes.c_void_p,  # group_gemm_sizes (GPU pointer)
        ctypes.c_void_p,  # g_lds (GPU pointer)
        ctypes.c_int,     # group_size
    ]

def get_cuda_torch_inputs_for_shape(shape: Tuple[int, ...]):
    """为特定形状创建CUDA和PyTorch实现的输入数据"""
    torch.manual_seed(SEED)

    group_size, M, N, K = shape

    # 创建矩阵组
    group_A = []
    group_B = []
    group_C_cuda = []

    for i in range(group_size):
        # 为每个矩阵创建数据
        a = torch.randn((M, K), dtype=torch.float16, device="cuda")
        b = torch.randn((K, N), dtype=torch.float16, device="cuda")
        c = torch.empty((M, N), dtype=torch.float16, device="cuda")

        group_A.append(a)
        group_B.append(b)
        group_C_cuda.append(c)

    # 准备设备端指针数组
    A_addrs = [a.data_ptr() for a in group_A]
    B_addrs = [b.data_ptr() for b in group_B]
    C_addrs = [c.data_ptr() for c in group_C_cuda]

    # 创建设备端张量
    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")

    # 创建尺寸和步长信息
    g_sizes = []
    g_lds = []
    for i in range(group_size):
        a, b, c = group_A[i], group_B[i], group_C_cuda[i]
        M_i, K_i = a.shape
        K_i, N_i = b.shape
        g_sizes.extend([M_i, N_i, K_i])
        g_lds.extend([a.stride(0), b.stride(0), c.stride(0)])

    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

    # CUDA输入：GPU指针列表
    cuda_all_inputs = [
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size
    ]

    # PyTorch输入：用于参考实现的张量列表
    torch_all_inputs = [group_A, group_B]

    # CUDA输出张量用于结果比较
    cuda_output_tensors = group_C_cuda

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

def torch_kernel(group_A: List[torch.Tensor], group_B: List[torch.Tensor]) -> List[torch.Tensor]:
    """PyTorch参考实现：grouped matrix multiplication"""
    group_C = []
    for A, B in zip(group_A, group_B):
        C = torch.matmul(A, B)
        group_C.append(C)
    return group_C