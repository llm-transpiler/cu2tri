import torch
import triton
import triton.language as tl
import operator

# Handle triton version compatibility for tanh
if triton.__version__ >= "3.0.0":
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    n_cols,
    n_non_ignore,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified Liger-Kernel CrossEntropy implementation
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Get label for this row
    label = tl.load(Y_ptr + row_idx * Y_stride)

    # Skip if label is ignore_index
    if label == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        return

    # Load logits row
    X_row = tl.load(X_ptr + row_idx * X_stride + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    # Compute logsumexp for numerical stability
    c = tl.max(X_row, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(X_row - c), 0))

    # Extract logit for target class
    target_logit = tl.load(X_ptr + row_idx * X_stride + label)

    # Cross entropy: logsumexp - target_logit
    loss = logsumexp - target_logit

    tl.store(loss_ptr + row_idx, loss)


def triton_kernel(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Triton实现：Simplified Cross Entropy Loss (基于Liger-Kernel)
    高效的交叉熵损失计算实现
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Ensure contiguous inputs
    logits = logits.contiguous()
    labels = labels.contiguous()

    # Flatten for processing
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    total_elements = logits_flat.shape[0]

    # Output tensor
    losses = torch.empty(total_elements, dtype=torch.float32, device=device)

    # Auto-tuning block size
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Cap at reasonable size

    # Launch kernel
    grid = lambda meta: (total_elements,)

    liger_cross_entropy_kernel[grid](
        logits_flat,
        logits_flat.stride(0),
        labels_flat,
        labels_flat.stride(0),
        losses,
        vocab_size,
        total_elements,  # Simplified - assumes no ignore tokens
        ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute mean (excluding ignore indices)
    valid_mask = labels_flat != ignore_index
    if valid_mask.sum() > 0:
        return losses[valid_mask].mean()
    else:
        return torch.tensor(0.0, device=device, dtype=torch.float32)