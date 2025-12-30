import torch
import triton
import triton.language as tl

from triton import next_power_of_2

def calculate_settings(n_cols):
    """Calculate optimal block size and warps based on tensor dimensions"""
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    return BLOCK_SIZE, num_warps

@triton.jit
def _kl_div_forward_kernel(
    pred_ptr, pred_row_stride,
    target_ptr, target_row_stride,
    loss_ptr, loss_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    KL Divergence forward kernel
    Computes: loss = target * (log(target) - pred)
    where pred is in log-space
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load predictions and targets
    pred_row = tl.load(pred_ptr + row_idx * pred_row_stride + col_offsets, mask=mask, other=0.0)
    target_row = tl.load(target_ptr + row_idx * target_row_stride + col_offsets, mask=mask, other=0.0)

    # Convert to fp32 for numerical stability
    pred_f32 = pred_row.to(tl.float32)
    target_f32 = target_row.to(tl.float32)
    eps_f32 = eps.to(tl.float32)

    # KL(target || pred) = target * (log(target) - pred)
    # pred is already in log-space
    loss = target_f32 * (tl.log(tl.maximum(target_f32, eps_f32)) - pred_f32)

    # Store result (cast back to original dtype)
    tl.store(loss_ptr + row_idx * loss_row_stride + col_offsets, loss.to(pred_row.dtype), mask=mask)

def triton_kernel(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    Triton实现：KL Divergence Loss
    高效的KL散度计算

    Args:
        predictions: Predictions in log-space (log probabilities)
        targets: Target probability distributions
        reduction: Reduction method ("none", "batchmean", "sum", "mean")

    Returns:
        KL Divergence loss
    """
    original_shape = predictions.shape
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Flatten all dimensions to 2D: (all_other_dims, vocab_size)
    n_cols = predictions.shape[-1]
    pred_2d = predictions.view(-1, n_cols)
    target_2d = targets.view(-1, n_cols)
    n_rows = pred_2d.shape[0]

    # Output tensor for per-element losses
    losses_2d = torch.empty_like(pred_2d)

    # Calculate optimal settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # Launch kernel
    with torch.cuda.device(predictions.device):
        _kl_div_forward_kernel[(n_rows,)](
            pred_2d,
            pred_2d.stride(0),
            target_2d,
            target_2d.stride(0),
            losses_2d,
            losses_2d.stride(0),
            n_cols,
            torch.tensor(1e-8, dtype=predictions.dtype, device=predictions.device),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    # Reshape back to original shape
    losses = losses_2d.view(original_shape)

    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    elif reduction == "batchmean":
        # Average over all dimensions except batch
        batch_size = original_shape[0]
        return losses.view(batch_size, -1).sum(dim=1).mean()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")