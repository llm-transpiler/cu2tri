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
def _softmax_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax forward kernel that handles both single and multi-block cases
    """
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load the row
    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-float("inf"))

    # Compute max for numerical stability
    m = tl.max(x, axis=0)

    # Compute exp and sum
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)

    # Compute softmax
    y = e / d

    # Store result
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask)

@triton.jit
def _softmax_multi_block_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multi-block softmax forward kernel for very large dimensions
    """
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: find global max
    m = -float("inf")
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"))
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.maximum(m, blk_max)
        m = new_m

    # Second pass: compute sum and final softmax
    d = 0.0
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"))
        d += tl.sum(tl.exp(xblk - m), axis=0)

    # Third pass: store results
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask, other=-float("inf"))
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask)

def triton_kernel(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton实现：Softmax
    高性能softmax激活函数实现，支持大维度
    """
    original_shape = input_tensor.shape
    input_tensor = input_tensor.contiguous()

    # Flatten all dimensions except the softmax dimension
    if dim != -1 and dim != len(original_shape) - 1:
        # Move softmax dimension to last
        input_tensor = input_tensor.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    # Reshape to 2D: (all_other_dims, softmax_dim)
    n_cols = input_tensor.shape[-1]
    input_2d = input_tensor.view(-1, n_cols)
    n_rows = input_2d.shape[0]

    # Output tensor
    output_2d = torch.empty_like(input_2d)

    # Calculate optimal settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # Launch kernel
    with torch.cuda.device(input_tensor.device):
        if n_cols <= BLOCK_SIZE:
            # Single block per row
            _softmax_forward_kernel[(n_rows,)](
                output_2d,
                output_2d.stride(0),
                input_2d,
                input_2d.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            # Multi-block for large dimensions
            _softmax_multi_block_forward_kernel[(n_rows,)](
                output_2d,
                output_2d.stride(0),
                input_2d,
                input_2d.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    # Reshape back to original shape
    output = output_2d.view(original_shape)

    # Transpose back if we transposed input
    if transposed:
        output = output.transpose(dim, -1)

    return output