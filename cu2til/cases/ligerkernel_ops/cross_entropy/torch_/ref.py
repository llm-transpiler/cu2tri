import torch

def torch_kernel(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    PyTorch参考实现：Cross Entropy Loss
    与Liger-Kernel实现保持一致
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross entropy
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='mean',
        ignore_index=ignore_index
    )

    return loss