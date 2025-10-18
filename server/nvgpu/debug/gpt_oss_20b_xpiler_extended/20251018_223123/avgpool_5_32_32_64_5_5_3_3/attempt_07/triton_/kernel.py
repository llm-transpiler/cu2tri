import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_H: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    output_size: tl.constexpr,
):
    global_idx = tl.program_id(0)
    blockIdx_x = global_idx // 1024
    threadIdx_x = global_idx % 1024

    B = blockIdx_x * 4 + (threadIdx_x >> 8)
    if B < 125:
        pool_sum = tl.zeros([], dtype=tl.float32)
        for rv0 in range(5):
            for rv1 in range(5):
                idx = (
                    (B // 25) * 65536
                    + (((((blockIdx_x * 8) + (threadIdx_x >> 7)) % 50) // 5) * 6144)
                    + (rv0 * 2048)
                    + ((((blockIdx_x * 6) + (threadIdx_x >> 6)) % 10) * 192)
                    + (rv1 * 64)
                    + (threadIdx_x & 63)
                )
                pool_sum += tl.load(input_ptr + idx)
        output_ptr[global_idx] = pool_sum * 0.04

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels
    grid = triton.cdiv(output_size, 1)
    _triton_kernel_impl[grid](
        input,
        output,
        batch_size,
        channels,
        input_H,
        kernel_size,
        stride,
        output_size,
        num_warps=32,
    )