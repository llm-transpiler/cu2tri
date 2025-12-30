import torch

def torch_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：Jensen-Shannon Divergence
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    其中 M = 0.5 * (P + Q)

    Args:
        p: First probability distribution
        q: Second probability distribution
        reduction: Reduction method ("none", "batchmean", "sum", "mean")

    Returns:
        Jensen-Shannon divergence
    """
    # Compute M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # Compute KL(P||M) and KL(Q||M)
    # KL(P||M) = sum(P * log(P/M)) = sum(P * (log(P) - log(M)))
    kl_pm = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + 1e-8) - torch.log(m + 1e-8)), dim=-1)

    # JS divergence
    jsd = 0.5 * (kl_pm + kl_qm)

    # Apply reduction
    if reduction == "none":
        return jsd
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    elif reduction == "batchmean":
        batch_size = p.shape[0]
        return jsd.view(batch_size, -1).mean(dim=1).mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")