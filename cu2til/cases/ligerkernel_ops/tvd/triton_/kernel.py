import torch
import triton
import triton.language as tl
from triton import next_power_of_2

def calculate_settings(n_cols):
    BLOCK_SIZE = next_power_of_2(n_cols)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
    return BLOCK_SIZE, num_warps

@triton.jit
def _tvd_forward_kernel(
    p_ptr, p_row_stride,
    q_ptr, q_row_stride,
    loss_ptr, loss_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """TVD(P||Q) = 0.5 * sum(|P - Q|)"""
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    p_row = tl.load(p_ptr + row_idx * p_row_stride + col_offsets, mask=mask, other=0.0)
    q_row = tl.load(q_ptr + row_idx * q_row_stride + col_offsets, mask=mask, other=0.0)

    # TVD = 0.5 * sum(|P - Q|)
    diff = tl.abs(p_row - q_row)
    tvd = 0.5 * tl.sum(diff, axis=0)

    tl.store(loss_ptr + row_idx * loss_row_stride, tvd)

def triton_kernel(p: torch.Tensor, q: torch.Tensor, reduction: str = "batchmean") -> torch.Tensor:
    """Triton实现：Total Variation Distance"""
    original_shape = p.shape
    p, q = p.contiguous(), q.contiguous()

    n_cols = p.shape[-1]
    p_2d, q_2d = p.view(-1, n_cols), q.view(-1, n_cols)
    n_rows = p_2d.shape[0]

    losses_1d = torch.zeros(n_rows, dtype=torch.float32, device=p.device)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with torch.cuda.device(p.device):
        _tvd_forward_kernel[(n_rows,)](
            p_2d, p_2d.stride(0), q_2d, q_2d.stride(0),
            losses_1d, losses_1d.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

    if reduction == "none":
        return losses_1d.view(original_shape[:-1])
    elif reduction == "sum":
        return losses_1d.sum()
    elif reduction == "mean":
        return losses_1d.mean()
    elif reduction == "batchmean":
        batch_size = original_shape[0]
        return losses_1d.view(batch_size, -1).mean(dim=1).mean()
    return losses_1d