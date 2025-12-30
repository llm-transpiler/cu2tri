import torch

def torch_kernel(logits: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """
    PyTorch参考实现：GRPO (Generalized Relative Policy Optimization) Loss
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute log_probs
    log_probs = torch.log_softmax(logits, dim=-1)

    # Sample action (use argmax for deterministic reference)
    actions = torch.argmax(probs, dim=-1)

    # Get action probabilities
    action_probs = torch.max(probs, dim=-1)[0]
    action_log_probs = torch.log(action_probs + 1e-8)

    # GRPO surrogate objective
    ratio = torch.exp(action_log_probs - action_log_probs.detach())  # Simplified
    surrogate = ratio * advantages

    # Clipped surrogate
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = clipped_ratio * advantages

    # GRPO loss: minimize negative of clipped objective
    loss = -torch.minimum(surrogate, clipped_surrogate).mean()

    return loss