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
def _poly_norm_forward_kernel(
    x_ptr, x_stride,
    y_ptr, y_stride,
    n_cols,
    alpha,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """Polynomial Normalization: y = (x + beta)^alpha"""
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row = tl.load(x_ptr + row_idx * x_stride + col_offsets, mask=mask, other=0.0)

    # y = (x + beta)^alpha
    x_shifted = x_row + beta
    x_clamped = tl.maximum(x_shifted, 1e-8)
    y_row = x_clamped ** alpha

    tl.store(y_ptr + row_idx * y_stride + col_offsets, y_row, mask=mask)

def triton_kernel(x: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """Triton实现：Polynomial Normalization"""
    original_shape = x.shape
    x = x.contiguous()

    n_cols = x.shape[-1]
    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.shape[0]

    y_2d = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with torch.cuda.device(x.device):
        _poly_norm_forward_kernel[(n_rows,)](
            x_2d, x_2d.stride(0),
            y_2d, y_2d.stride(0),
            n_cols, alpha, beta,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

    return y_2d.view(original_shape)