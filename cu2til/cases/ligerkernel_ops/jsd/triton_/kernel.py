import torch
import triton
import triton.language as tl

from triton import next_power_of_2

def calculate_settings(n_cols):
    """Calculate optimal block size and warps"""
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    return BLOCK_SIZE, num_warps

@triton.jit
def _jsd_forward_kernel(
    p_ptr, p_row_stride,
    q_ptr, q_row_stride,
    loss_ptr, loss_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Jensen-Shannon Divergence forward kernel
    Computes: JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load probabilities
    p_row = tl.load(p_ptr + row_idx * p_row_stride + col_offsets, mask=mask, other=0.0)
    q_row = tl.load(q_ptr + row_idx * q_row_stride + col_offsets, mask=mask, other=0.0)

    # Convert to fp32 for stability
    p_f32 = p_row.to(tl.float32)
    q_f32 = q_row.to(tl.float32)
    eps_f32 = eps.to(tl.float32)

    # Compute M = 0.5 * (P + Q)
    m = 0.5 * (p_f32 + q_f32)

    # Compute KL(P||M) = sum(P * (log(P) - log(M)))
    kl_pm = p_f32 * (tl.log(tl.maximum(p_f32, eps_f32)) - tl.log(tl.maximum(m, eps_f32)))

    # Compute KL(Q||M) = sum(Q * (log(Q) - log(M)))
    kl_qm = q_f32 * (tl.log(tl.maximum(q_f32, eps_f32)) - tl.log(tl.maximum(m, eps_f32)))

    # JS = 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (kl_pm + kl_qm)

    # Sum across columns and store result
    loss = tl.sum(jsd, axis=0)
    tl.store(loss_ptr + row_idx * loss_row_stride, loss.to(p_row.dtype))

def triton_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """
    Triton实现：Jensen-Shannon Divergence
    高效的JS散度计算
    """
    original_shape = p.shape
    p = p.contiguous()
    q = q.contiguous()

    # Flatten to 2D
    n_cols = p.shape[-1]
    p_2d = p.view(-1, n_cols)
    q_2d = q.view(-1, n_cols)
    n_rows = p_2d.shape[0]

    # Output tensor
    losses_1d = torch.zeros(n_rows, dtype=torch.float32, device=p.device)

    # Calculate settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # Launch kernel
    with torch.cuda.device(p.device):
        _jsd_forward_kernel[(n_rows,)](
            p_2d,
            p_2d.stride(0),
            q_2d,
            q_2d.stride(0),
            losses_1d,
            losses_1d.stride(0),
            n_cols,
            torch.tensor(1e-8, dtype=p.dtype, device=p.device),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    # Apply reduction
    if reduction == "none":
        return losses_1d.to(p.dtype).view(original_shape[:-1])
    elif reduction == "sum":
        return losses_1d.to(p.dtype).sum()
    elif reduction == "mean":
        return losses_1d.to(p.dtype).mean()
    elif reduction == "batchmean":
        batch_size = original_shape[0]
        return losses_1d.to(p.dtype).view(batch_size, -1).mean(dim=1).mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")