import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    num_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id(0)
    total = batch_size * num_channels * spatial_size
    if idx >= total:
        return

    channel = (idx // spatial_size) % num_channels

    x = tl.load(input_ptr + idx)
    mu = tl.load(mean_ptr + channel)
    var_val = tl.load(var_ptr + channel)
    gamma_val = tl.load(gamma_ptr + channel)
    beta_val = tl.load(beta_ptr + channel)

    norm = (x - mu) / tl.math.sqrt(var_val + 1e-5)
    y = norm * gamma_val + beta_val

    tl.store(output_ptr + idx, y)

def triton_kernel(
    input,
    output,
    mean,
    var,
    gamma,
    beta,
    batch_size,
    num_channels,
    spatial_size,
):
    total = batch_size * num_channels * spatial_size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    _triton_kernel_impl[grid](
        input,
        output,
        mean,
        var,
        gamma,
        beta,
        batch_size,
        num_channels,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )