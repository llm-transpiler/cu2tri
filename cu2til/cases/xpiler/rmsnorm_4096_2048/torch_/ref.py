import torch

def torch_kernel(*args):
    """PyTorch参考实现 for rmsnorm"""
    x = args[0]
    # RMS Norm implementation
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1e-6)
