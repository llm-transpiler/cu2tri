import torch
import torch.nn.functional as F

def torch_kernel(_input: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                bias: torch.Tensor = None, ignore_index: int = -100, reduction: str = "mean") -> torch.Tensor:
    """
    PyTorch参考实现：Fused Linear Cross Entropy
    先计算线性变换，然后计算交叉熵损失

    Args:
        _input: Input tensor of shape (batch_tokens, hidden_dim)
        weight: Weight tensor of shape (vocab_size, hidden_dim)
        target: Target tensor of shape (batch_tokens,) with values in [0, vocab_size-1]
        bias: Optional bias tensor of shape (vocab_size,)
        ignore_index: Target index to ignore
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        Cross entropy loss tensor
    """
    # Linear forward: logits = _input @ weight.t()
    logits = torch.matmul(_input, weight.t())
    if bias is not None:
        logits = logits + bias

    # Compute cross entropy loss
    loss = F.cross_entropy(
        logits, target,
        ignore_index=ignore_index,
        reduction=reduction
    )

    return loss