import torch

def torch_kernel(*args):
    """PyTorch参考实现 for gemv"""
    A, x = args[0], args[1]
    return torch.matmul(A, x)
