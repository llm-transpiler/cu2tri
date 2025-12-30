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
def _fused_add_rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    S_ptr, S_row_stride,
    X_ptr, X_row_stride,
    R_ptr, R_row_stride,
    W_ptr, W_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Add RMS Norm forward kernel
    Computes:
    1. S = X + R (add residual)
    2. Store S as new residual
    3. Y = RMSNorm(S) * W
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input and residual
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + row_idx * R_row_stride + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # Step 1: Add residual
    S_row = X_row + R_row

    # Step 2: Store the sum as new residual
    tl.store(S_ptr + row_idx * S_row_stride + col_offsets, S_row, mask=mask)

    # Step 3: RMS Normalization
    # Convert to fp32 for numerical stability
    S_row_f32 = S_row.to(tl.float32)
    eps_f32 = eps.to(tl.float32)

    # Compute RMS: sqrt(mean(square(x)))
    mean_square = tl.sum(S_row_f32 * S_row_f32, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps_f32)

    # Normalize
    S_row_normalized = S_row_f32 * rstd

    # Apply weight and cast back to original dtype
    Y_row = S_row_normalized * W_row
    Y_row = Y_row.to(S_row.dtype)

    # Store result
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)

def triton_kernel(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                 eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton实现：Fused Add RMS Norm
    融合加法残差和RMS归一化，减少内存访问
    """
    original_shape = hidden_states.shape
    hidden_states = hidden_states.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    # Flatten all dimensions to 2D: (all_other_dims, hidden_dim)
    n_cols = hidden_states.shape[-1]
    hidden_states_2d = hidden_states.view(-1, n_cols)
    residual_2d = residual.view(-1, n_cols)
    n_rows = hidden_states_2d.shape[0]

    # Output tensors
    normalized_2d = torch.empty_like(hidden_states_2d)
    new_residual_2d = torch.empty_like(residual_2d)

    # Calculate optimal settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # Launch kernel
    with torch.cuda.device(hidden_states.device):
        _fused_add_rms_norm_forward_kernel[(n_rows,)](
            normalized_2d,
            normalized_2d.stride(0),
            new_residual_2d,
            new_residual_2d.stride(0),
            hidden_states_2d,
            hidden_states_2d.stride(0),
            residual_2d,
            residual_2d.stride(0),
            weight,
            weight.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    # Reshape back to original shape
    normalized = normalized_2d.view(original_shape)
    new_residual = new_residual_2d.view(original_shape)

    return normalized, new_residual