import torch

def torch_kernel(gate: torch.Tensor, up: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    """
    PyTorch参考实现：GEGLU (Gated Exponential Linear Unit)
    GEGLU(gate, up) = Swish(gate) * up

    Args:
        gate: Gate projection tensor
        up: Up projection tensor
        approximate: Whether to use tanh approximation (True) or exact erf (False)

    Returns:
        GEGLU output tensor
    """
    if approximate:
        # Approximate version using tanh approximation
        # Swish(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x²)))
        x = gate
        s = 0.7978845608028654  # sqrt(2/π)
        swish = 0.5 * x * (1 + torch.tanh(s * x * (1 + 0.044715 * x * x)))
    else:
        # Exact version using error function
        # Swish(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
        x = gate
        swish = 0.5 * x * (1 + torch.erf(x / (2**0.5)))

    return swish * up