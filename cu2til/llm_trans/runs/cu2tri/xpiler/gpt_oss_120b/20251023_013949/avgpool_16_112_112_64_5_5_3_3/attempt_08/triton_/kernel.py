import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, pool_avg, output_size, input_size, BLOCK_SIZE: tl.constexpr):
    # Program (block) ID
    pid = tl.program_id(0)
    # Thread indices within the block
    t = tl.arange(0, BLOCK_SIZE)
    # Global output index for each thread
    out_idx = pid * BLOCK_SIZE + t
    mask_out = out_idx < output_size

    # Alias for readability
    b = pid
    thread_idx = t

    # --- Compute input address components (mirroring the original CUDA indexing) ---
    # Batch offset
    b_div81 = b // 81
    term0 = b_div81 * 802816                     # batch_id * (C*H*W)

    # Output height (oh) component
    b_mod81 = b % 81
    term1 = ((b_mod81 * 4 + (thread_idx >> 8)) // 9) * 21504  # oh * (H*C*stride)

    # Output width (ow) component
    term3 = ((b * 16 + (thread_idx >> 6)) % 36) * 192          # ow * (C*stride)

    # Channel component (c)
    term5 = thread_idx & 63                                   # c (0..63)

    # Accumulate sum over the 5×5 window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        term2 = rv0 * 7168                 # rv0 * (H*C)
        for rv1 in range(5):
            term4 = rv1 * 64               # rv1 * C
            idx = term0 + term1 + term2 + term3 + term4 + term5
            mask_in = idx < input_size
            a = tl.load(A + idx, mask=mask_in & mask_out, other=0.0)
            sum_val += a

    # Write the average (1/25 = 0.04)
    avg = sum_val * 0.04
    tl.store(pool_avg + out_idx, avg, mask=mask_out)

def triton_kernel(input: torch.Tensor,
                  output: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Triton implementation of the 5×5 average‑pooling kernel.
    Signature matches the original CUDA kernel.
    """
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    # Total number of output elements (flattened NHWC)
    output_size = batch_size * output_H * output_H * channels
    # Total number of input elements (flattened NHWC)
    input_size = batch_size * channels * input_H * input_H

    BLOCK_SIZE = 1024
    # Grid size must be a tuple for Triton
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32
    )