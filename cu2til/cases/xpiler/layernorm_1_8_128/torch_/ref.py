import torch

def torch_kernel(*args):
    """PyTorch参考实现 for layernorm"""
    x = args[0]
    return torch.nn.functional.layer_norm(x, x.shape[-1:])
