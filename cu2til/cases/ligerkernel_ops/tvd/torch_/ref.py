import torch

def torch_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    PyTorch参考实现：Total Variation Distance
    TVD(P||Q) = 0.5 * sum(|P - Q|)
    """
    tvd = 0.5 * torch.sum(torch.abs(p - q), dim=-1)

    if reduction == "none":
        return tvd
    elif reduction == "sum":
        return tvd.sum()
    elif reduction == "mean":
        return tvd.mean()
    elif reduction == "batchmean":
        batch_size = p.shape[0]
        return tvd.view(batch_size, -1).mean(dim=1).mean()
    return tvd