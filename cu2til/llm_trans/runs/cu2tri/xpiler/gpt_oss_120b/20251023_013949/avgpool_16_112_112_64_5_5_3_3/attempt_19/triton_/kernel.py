import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: functional replica of the original CUDA average‑pooling kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,               # *float32 input tensor (flattened)
    pool_avg_ptr,        # *float32 output tensor (flattened)
    N: tl.int64,         # total number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    # Program (block) and thread indices
    pid = tl.program_id(0).to(tl.int64)                     # blockIdx.x (scalar)
    tid = tl.arange(0, BLOCK_SIZE).to(tl.int64)             # threadIdx.x (vector)

    # Global output index and mask for out‑of‑bounds threads
    out_idx = pid * BLOCK_SIZE + tid
    mask = out_idx < N

    # ------------------------------------------------------------------
    # Recreate the exact indexing pattern from the CUDA kernel
    # ------------------------------------------------------------------
    outer_idx = pid // 81                                   # (int)blockIdx.x / 81
    tile_idx  = pid % 81                                    # (int)blockIdx.x % 81

    t0 = tid // 256                                         # (int)threadIdx.x >> 8
    t1 = tid // 64                                          # (int)threadIdx.x >> 6
    t2 = tid % 64                                           # (int)threadIdx.x & 63

    # Base offset for the top‑left element of the 5×5 window
    base_offset = (
        outer_idx * 802816
        + ((tile_idx * 4 + t0) // 9) * 21504
        + ((pid * 16 + t1) % 36) * 192
        + t2
    )

    # ------------------------------------------------------------------
    # Accumulate sum over the 5×5 pooling region
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):        # vertical offset
        for rv1 in range(5):    # horizontal offset
            offset = base_offset + rv0 * 7168 + rv1 * 64
            sum_val += tl.load(A_ptr + offset, mask=mask)

    # ------------------------------------------------------------------
    # Compute average (scale factor 1/25 = 0.04) and write back
    # ------------------------------------------------------------------
    avg = sum_val * 0.04
    tl.store(pool_avg_ptr + out_idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper matching the original CUDA kernel signature
# ----------------------------------------------------------------------
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
    Triton implementation of the average‑pooling kernel.
    Signature mirrors the original CUDA kernel:
        (float *input, float *output,
         int batch_size, int channels,
         int input_H, int kernel_size, int stride)
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "Tensors must be CUDA tensors"
    input = input.contiguous()
    output = output.contiguous()

    # Compute output dimensions (identical to the CUDA host code)
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Launch configuration
    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Kernel launch (32 warps = 1024 threads per block)
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )