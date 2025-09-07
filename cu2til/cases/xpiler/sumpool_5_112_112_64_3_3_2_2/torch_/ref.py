import torch

def torch_kernel(*args):
    """PyTorch参考实现 for sumpool"""
    x = args[0]
    return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2) * (3 * 3)
