import torch
import triton
import triton.language as tl

# Triton kernel implementing the same logic as the original CUDA kernel.
# The kernel computes a 5x5 average pooling (scale factor 1/25 = 0.04).
# It assumes the same data layout and dimensions as the CUDA version.
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # pointer to input tensor (float32)
    pool_avg_ptr,        # pointer to output tensor (float32)
    BLOCK_SIZE: tl.constexpr,  # number of threads per block (1024)
):
    # Program (block) and thread indices
    pid = tl.program_id(0)                     # blockIdx.x
    tid = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)  # threadIdx.x

    # Decompose block and thread indices to match the CUDA indexing
    outer_idx = pid // 81                       # (int)blockIdx.x / 81
    tile_idx  = pid % 81                        # (int)blockIdx.x % 81

    t0 = tid // 256                             # (int)threadIdx.x >> 8
    t1 = tid // 64                              # (int)threadIdx.x >> 6
    t2 = tid % 64                               # (int)threadIdx.x & 63

    # Base offset for the top‑left element of the 5×5 window (no rv0/rv1 yet)
    base_offset = (
        outer_idx * 802816
        + ((tile_idx * 4 + t0) // 9) * 21504
        + ((pid * 16 + t1) % 36) * 192
        + t2
    )

    # Accumulate the sum over the 5×5 region
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):        # vertical offset
        for rv1 in range(5):    # horizontal offset
            offset = base_offset + rv0 * 7168 + rv1 * 64
            sum_val += tl.load(A_ptr + offset, dtype=tl.float32)

    # Compute the average (scale by 1/25 = 0.04)
    avg = sum_val * 0.04

    # Write the result to the output tensor
    out_offset = pid * BLOCK_SIZE + tid
    tl.store(pool_avg_ptr + out_offset, avg, dtype=tl.float32)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Wrapper that launches the Triton average‑pooling kernel.
    The signature matches the original CUDA kernel.
    """
    # Ensure tensors are on CUDA and contiguous
    assert input.is_cuda and output.is_cuda, "Tensors must be CUDA tensors"
    input = input.contiguous()
    output = output.contiguous()

    # Compute output dimensions (same as the CUDA host code)
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
    )