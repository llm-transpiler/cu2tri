import torch
import torch.nn.functional as F

def torch_kernel(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：KL Divergence Loss
    计算 KL(P||Q) = sum(P * (log(P) - log(Q)))

    Args:
        predictions: Predictions in log-space (log probabilities)
        targets: Target probability distributions (NOT in log-space)
        reduction: Reduction method ("none", "batchmean", "sum", "mean")

    Returns:
        KL Divergence loss
    """
    # PyTorch's kl_div expects:
    # - input: log-probabilities
    # - target: probabilities (by default) or log-probabilities if log_target=True
    loss = F.kl_div(predictions, targets, reduction=reduction, log_target=False)
    return loss