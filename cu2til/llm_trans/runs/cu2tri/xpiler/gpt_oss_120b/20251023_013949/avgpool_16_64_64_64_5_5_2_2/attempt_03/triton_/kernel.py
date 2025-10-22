import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    tid = tl.arange(0, BLOCK_SIZE).to(tl.int32)

    # Global linear index for the output element
    global_idx = pid * BLOCK_SIZE + tid
    mask = global_idx < output_size

    # Reproduce the exact indexing from the original CUDA kernel
    term1 = ((pid * 4 + (tid >> 8)) // 225) * 262144
    term2 = (((pid * 8 + (tid >> 7)) % 450) // 15) * 8192
    term4 = ((pid * 16 + (tid >> 6)) % 30) * 128
    base_offset = term1 + term2 + term4 + (tid & 63)

    # Accumulate the 5×5 window
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        offset_rv0 = rv0 * 4096
        for rv1 in range(5):
            offset = base_offset + offset_rv0 + rv1 * 64
            pool_sum += tl.load(A_ptr + offset, mask=mask, other=0.0)

    # Write the average (1/25 = 0.04)
    tl.store(pool_avg_ptr + global_idx, pool_sum * 0.04, mask=mask)

def triton_kernel(input: torch.Tensor, output: torch.Tensor,
                  batch_size: int, channels: int,
                  input_H: int, kernel_size: int, stride: int):
    """
    Triton implementation of the CUDA average‑pooling kernel.
    Signature matches the original CUDA kernel.
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 tensors are supported"

    # Compute output dimensions
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel
    _triton_kernel_impl[grid](input, output, output_size, BLOCK_SIZE=BLOCK_SIZE)