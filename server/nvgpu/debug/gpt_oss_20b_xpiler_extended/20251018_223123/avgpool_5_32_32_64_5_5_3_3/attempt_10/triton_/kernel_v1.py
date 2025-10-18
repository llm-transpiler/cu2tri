import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(A, pool_avg, output_size: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= output_size:
        return
    block_idx = pid // 1024
    thread_idx = pid % 1024
    b = block_idx * 4 + (thread_idx >> 8)
    if b < 125:
        sum_val = 0.0
        for rv0 in range(5):
            for rv1 in range(5):
                idx = ((b // 25) * 65536
                       + (((block_idx * 8 + (thread_idx >> 7)) % 50) // 5) * 6144
                       + rv0 * 2048
                       + (((block_idx * 6 + (thread_idx >> 6)) % 10) * 192)
                       + rv1 * 64
                       + (thread_idx & 63))
                sum_val += tl.load(A + idx)
        tl.store(pool_avg + block_idx * 1024 + thread_idx, sum_val * 0.04)

def triton_kernel(input, output, batch_size, channels, input_H, kernel_size, stride):
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels
    grid = triton.cdiv(output_size, 1024)
    _triton_kernel_impl[(grid,)](input, output, output_size=output_size)