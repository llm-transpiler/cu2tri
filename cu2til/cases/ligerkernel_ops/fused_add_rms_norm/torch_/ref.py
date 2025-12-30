import torch

def torch_kernel(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现：Fused Add RMS Norm
    执行以下操作：
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states (after addition)
    3. hidden_states = rmsnorm(hidden_states)

    Args:
        hidden_states: Input hidden states tensor
        residual: Residual tensor to add
        weight: Weight tensor for RMS norm
        eps: Small value for numerical stability

    Returns:
        Tuple of (normalized_hidden_states, updated_residual)
    """
    # Step 1: Add residual to hidden states
    hidden_states = hidden_states + residual

    # Step 2: Update residual to be the sum
    residual = hidden_states.clone()

    # Step 3: Apply RMS normalization
    # Compute RMS: sqrt(mean(square(x)))
    rms = torch.rsqrt(torch.mean(hidden_states.pow(2), dim=-1, keepdim=True) + eps)

    # Normalize and apply weight
    normalized_hidden_states = hidden_states * rms * weight

    return normalized_hidden_states, residual