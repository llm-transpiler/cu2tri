import torch
import torch.nn.functional as F

def torch_kernel(logits: torch.Tensor, labels: torch.Tensor, logit_softcapping: torch.Tensor, logit_scaling: torch.Tensor) -> torch.Tensor:
    """
    PyTorch参考实现：Cross Entropy Loss
    与unsloth实现保持一致的预处理和后处理
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Extract scalar values
    softcap = logit_softcapping.item()
    scale = logit_scaling.item()

    # Apply logit scaling if specified (Cohere style)
    if scale != 1.0:
        logits = logits * scale

    # Apply logit softcapping if specified (Gemma 2 style)
    if softcap != 0.0:
        # x_cap = t * tanh(x / t)
        logits = softcap * torch.tanh(logits / softcap)

    # Reshape for cross entropy computation
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy loss with proper masking
    loss_flat = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
        ignore_index=-100
    )

    # Reshape back
    loss = loss_flat.view(batch_size, seq_len)

    # Compute mean over non-padding tokens
    n_items = (labels != -100).sum()
    if n_items > 0:
        return loss.sum() / n_items
    else:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)