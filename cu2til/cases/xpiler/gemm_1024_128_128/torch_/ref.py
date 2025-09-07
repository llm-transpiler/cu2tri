import torch

def torch_kernel(*args):
    """PyTorch参考实现 for gemm"""
    A, B = args[0], args[1]
    return torch.matmul(A.float(), B.float())
