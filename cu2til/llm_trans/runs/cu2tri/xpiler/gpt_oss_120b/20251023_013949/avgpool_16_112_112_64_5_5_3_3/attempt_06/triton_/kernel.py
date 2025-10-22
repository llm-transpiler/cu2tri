import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    # Program ID (block index) and thread offsets
    pid = tl.program_id(0)               # blockIdx.x
    offsets = tl.arange(0, BLOCK_SIZE)    # threadIdx.x

    # Linear output index for each thread
    out_idx = pid * BLOCK_SIZE + offsets

    # Mask for valid output elements (avoid OOB accesses)
    valid = out_idx < output_size

    # Accumulator for the sum of the 5x5 window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Pre‑compute parts of the input index that do not depend on rv0/rv1
    i0 = (pid // 81) * 802816
    i1 = ((((pid % 81) * 4 + (offsets >> 8)) // 9) * 21504)
    i3 = ((((pid * 16) + (offsets >> 6)) % 36) * 192)
    i5 = offsets & 63

    # Loop over the 5×5 pooling window
    for rv0 in tl.static_range(5):
        i2 = rv0 * 7168
        for rv1 in tl.static_range(5):
            i4 = rv1 * 64
            idx = i0 + i1 + i2 + i3 + i4 + i5
            a = tl.load(A_ptr + idx, mask=valid, other=0.0)
            sum_val += a

    # Write the average (sum * 1/25) to the output
    tl.store(pool_avg_ptr + out_idx, sum_val * 0.04, mask=valid)


def triton_kernel(input: torch.Tensor, output: torch.Tensor,
                  batch_size: int, channels: int, input_H: int,
                  kernel_size: int, stride: int):
    """
    Triton implementation of the average‑pooling kernel.
    The signature matches the original CUDA host function:
        cuda_kernel(float *input, float *output,
                    int batch_size, int channels,
                    int input_H, int kernel_size, int stride)

    Note:
        The kernel is specialized for a 5×5 window and a stride of 3,
        which matches the indexing pattern in the original CUDA code.
        Passing a different stride will produce incorrect results.
    """
    # Basic validation
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("input and output must be torch.float32")
    # Ensure contiguous layout (NHWC)
    input = input.contiguous()
    output = output.contiguous()

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        input,
        output,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Synchronize to make the kernel effects visible
    torch.cuda.synchronize()