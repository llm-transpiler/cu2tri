import torch

def torch_kernel(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch参考实现：Layer Normalization
    与Liger-Kernel实现保持一致
    """
    return torch.nn.functional.layer_norm(X, X.shape[-1:], weight=W, bias=B, eps=eps)