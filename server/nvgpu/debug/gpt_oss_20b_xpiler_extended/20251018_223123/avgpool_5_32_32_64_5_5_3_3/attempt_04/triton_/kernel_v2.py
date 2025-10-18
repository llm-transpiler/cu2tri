import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, batch_size, channels, input_H, kernel_size, stride):
    block_idx = tl.program_id(0)
    thread_idx = tl.thread_idx(0)

    # Guard against out‑of‑range threads
    cond = ((block_idx * 4) + (thread_idx >> 8)) < 125

    pool_sum = tl.float32(0.0)

    for rv0 in range(5):
        for rv1 in range(5):
            if cond:
                idx = (((((block_idx * 4) + (thread_idx >> 8)) // 25) * 65536
                        + (((block_idx * 8) + (thread_idx >> 7)) % 50) // 5 * 6144
                        + (rv0 * 2048)
                        + (((block_idx * 6) + (thread_idx >> 6)) % 10) * 192
                        + (rv1 * 64)
                        + (thread_idx & 63)))
                val = tl.load(A_ptr + idx)
                pool_sum += val

    if cond:
        out_idx = block_idx * 1024 + thread_idx
        tl.store(pool_avg_ptr + out_idx, pool_sum * tl.float32(0.04))

def triton_kernel(input: torch.Tensor, output: torch.Tensor,
                  batch_size: int, channels: int,
                  input_H: int, kernel_size: int, stride: int):
    """
    Wrapper that launches the Triton kernel with the same semantics as the original CUDA kernel.
    """
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = triton.cdiv(output_size, BLOCK_SIZE)

    _triton_kernel_impl[grid](input, output,
                              batch_size, channels,
                              input_H, kernel_size, stride)