import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    Y_ptr,
    Y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS Normalization kernel
    y_i = (x_i / RMS) * w_i, where RMS = sqrt(sum(x_i^2) / N + eps)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_row_stride
    Y_ptr += row_idx * Y_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # Convert to float32 for numerical stability
    X_row_f32 = X_row.to(tl.float32)
    W_row_f32 = W_row.to(tl.float32)

    # Compute RMS - sum over all elements in the row
    mean_square = tl.sum(X_row_f32 * X_row_f32) / n_cols.to(tl.float32)
    rstd = tl.rsqrt(mean_square + eps)

    # Apply RMS norm
    Y_row = X_row_f32 * rstd * W_row_f32

    # Store result (cast back to original dtype)
    tl.store(Y_ptr + col_offsets, Y_row.to(X_row.dtype), mask=mask)

def triton_kernel(hidden_states: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                  eps: float, batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    """
    Triton实现：RMS normalization
    """
    # Ensure inputs are contiguous
    hidden_states = hidden_states.contiguous()
    weight = weight.contiguous()

    original_shape = hidden_states.shape

    # Reshape to 2D: [batch_size * seq_len, hidden_size]
    hidden_states_2d = hidden_states.view(-1, hidden_size)
    output_2d = output.view(-1, hidden_size)

    n_rows = hidden_states_2d.shape[0]
    BLOCK_SIZE = 1024
    grid = (n_rows,)

    # Launch kernel
    rms_norm_kernel[grid](
        X_ptr=hidden_states_2d,
        X_row_stride=hidden_states_2d.stride(0),
        W_ptr=weight,
        W_row_stride=weight.stride(0),
        Y_ptr=output_2d,
        Y_row_stride=output_2d.stride(0),
        n_cols=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

def triton_rms_norm(hidden_states, weight, eps=1e-6):
    """保持向后兼容的函数名"""
    return torch_kernel(hidden_states, weight, eps)

def torch_kernel(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch参考实现：RMS normalization"""
    # 计算RMS: sqrt(mean(x^2) + eps)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    rms = torch.rsqrt(variance + eps)

    # 应用RMS norm
    return hidden_states * rms * weight