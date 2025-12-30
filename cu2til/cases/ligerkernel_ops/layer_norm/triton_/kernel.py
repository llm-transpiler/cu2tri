import torch
import triton
import triton.language as tl
import operator

# Handle triton version compatibility for rsqrt
if triton.__version__ >= "3.0.0":
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt

@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,              # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,        # stride of each row in output
    X_ptr,              # pointer to input, shape (n_rows, n_cols)
    X_row_stride,        # stride of each row in input
    W_ptr,              # pointer to weights, shape (n_cols,)
    W_row_stride,        # stride of each row in weights
    B_ptr,              # pointer to bias, shape (n_cols,)
    B_row_stride,        # stride of each row in bias
    Mean_ptr,            # pointer to mean, shape (n_rows,)
    Mean_row_stride,     # stride of each row in mean
    RSTD_ptr,            # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,     # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Liger-Kernel LayerNorm forward implementation
    Based on https://arxiv.org/abs/1607.06450
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pre-load weights and bias in fp32 for numerical stability
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)

    # Calculate pointers for this row
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    # Load input data and convert to fp32 for numerical stability
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    # Compute statistics in fp32 for numerical stability
    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean

    # Apply mask to variance calculation
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = rsqrt(var + eps)

    # Store statistics
    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))

    # Fused normalization and affine transformation
    # Y = (X - mean) * rstd * W + B
    Y_f32 = X_centered * rstd * W_f32 + B_f32

    # Store output
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


def triton_kernel(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton实现：Layer Normalization (基于Liger-Kernel)
    高性能LayerNorm实现，支持权重和偏置
    """
    batch_size, hidden_size = X.shape
    device = X.device

    # Ensure contiguous inputs
    X = X.contiguous()
    W = W.contiguous()
    B = B.contiguous()

    # Output tensor
    Y = torch.empty_like(X, dtype=X.dtype, device=device)

    # Intermediate tensors for statistics
    Mean = torch.empty(batch_size, dtype=X.dtype, device=device)
    RSTD = torch.empty(batch_size, dtype=X.dtype, device=device)

    # Auto-tuning block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # Cap at reasonable size

    # Launch kernel
    grid = lambda meta: (batch_size,)

    _layer_norm_forward_kernel[grid](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        B,
        B.stride(0),
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Y