import torch

def torch_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现：SwiGLU (Swish-Gated Linear Unit)
    SwiGLU(gate, up) = Swish(gate) * up where Swish(x) = x * sigmoid(x)
    """
    # Swish activation: f = gate * sigmoid(gate)
    swish = gate * torch.sigmoid(gate)
    # SwiGLU: h = swish * up
    return swish * up