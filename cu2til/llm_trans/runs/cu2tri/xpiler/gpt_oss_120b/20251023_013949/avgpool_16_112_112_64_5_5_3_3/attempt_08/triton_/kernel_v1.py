import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, pool_avg, output_size, input_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # thread indices within the block
    t = tl.arange(0, BLOCK_SIZE)
    # global output index
    out_idx = pid * BLOCK_SIZE + t
    mask_out = out_idx < output_size

    # blockIdx.x analog
    b = pid
    # threadIdx.x analog
    thread_idx = t

    # compute components of the input index (direct translation of the CUDA expression)
    b_div81 = b // 81
    b_mod81 = b % 81

    term0 = b_div81 * 802816
    term1 = ((b_mod81 * 4 + (thread_idx >> 8)) // 9) * 21504
    term3 = ((b * 16 + (thread_idx >> 6)) % 36) * 192
    term5 = thread_idx & 63

    # accumulate the sum over the 5×5 window
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        term2 = rv0 * 7168
        for rv1 in range(5):
            term4 = rv1 * 64
            idx = term0 + term1 + term2 + term3 + term4 + term5
            # guard against out‑of‑bounds reads
            mask_in = idx < input_size
            a = tl.load(A + idx, mask=mask_in & mask_out, other=0.0)
            sum_val += a

    # write the average (1/25 = 0.04)
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
    The signature matches the original CUDA kernel.
    """
    # compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    # total number of output elements
    output_size = batch_size * output_H * output_H * channels
    # total number of input elements (N·C·H·W)
    input_size = batch_size * channels * input_H * input_H

    BLOCK_SIZE = 1024
    # grid size = ceil(output_size / BLOCK_SIZE)
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32  # 1024 threads = 32 warps
    )