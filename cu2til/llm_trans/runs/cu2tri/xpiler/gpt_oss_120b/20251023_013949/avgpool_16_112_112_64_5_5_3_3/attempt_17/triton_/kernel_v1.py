import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: average pooling over a 5x5 window.
# The implementation reproduces the exact indexing logic of the original CUDA
# kernel (including all hard‑coded strides).  It assumes the same tensor layout
# as the CUDA version (batch stride = 802816, etc.).
# ---------------------------------------------------------------------------

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel.
    Parameters
    ----------
    A_ptr : pointer
        Input tensor (float32) – flattened.
    pool_avg_ptr : pointer
        Output tensor (float32) – flattened.
    output_size : int
        Total number of output elements (batch * out_h * out_w * channels).
    BLOCK_SIZE : int (constexpr)
        Number of threads per program (matches CUDA blockDim.x = 1024).
    """
    pid = tl.program_id(0)               # corresponds to blockIdx.x
    offs = tl.arange(0, BLOCK_SIZE)      # corresponds to threadIdx.x

    # Global linear index for the output element this thread is responsible for
    out_idx = pid * BLOCK_SIZE + offs
    mask = out_idx < output_size

    # -----------------------------------------------------------------------
    # Recreate the exact address calculation from the CUDA kernel.
    # The constants (802816, 21504, 7168, 192, 64) are derived from the
    # original tensor dimensions (e.g., channels=256, H=W=56).
    # -----------------------------------------------------------------------
    pid_div_81 = pid // 81
    pid_mod_81 = pid % 81

    term1 = pid_div_81 * 802816                     # batch stride
    term2 = ((pid_mod_81 * 4 + (offs >> 8)) // 9) * 21504
    term4 = ((pid * 16 + (offs >> 6)) % 36) * 192
    term6 = offs & 63

    base_idx = term1 + term2 + term4 + term6

    # Accumulate the sum over the 5×5 window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        for rv1 in range(5):
            idx = base_idx + rv0 * 7168 + rv1 * 64
            sum_val += tl.load(A_ptr + idx, mask=mask, other=0.0)

    # Compute the average (divide by 25)
    avg = sum_val * 0.04  # 1/25

    # Write the result
    tl.store(pool_avg_ptr + out_idx, avg, mask=mask)


def triton_kernel(input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Entry‑point that mirrors the original `cuda_kernel` signature.
    Launches the Triton kernel with the same grid configuration
    (block size = 1024 threads) and computes the required output size.
    """
    # Basic sanity checks
    assert input_tensor.is_cuda and output_tensor.is_cuda, \
        "Both input and output tensors must reside on the CUDA device."
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, \
        "Only float32 tensors are supported."

    # Compute output spatial dimension and total number of output elements
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024  # matches the original CUDA blockDim.x
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32  # 1024 threads = 32 warps
    )