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
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    lane = tl.arange(0, BLOCK_SIZE, dtype=tl.int32)
    idx = block_start + lane

    total = batch_size * num_channels * spatial_size
    mask = idx < tl.int32(total)

    input_val = tl.load(input_ptr + idx, mask=mask, other=0.0)

    channel = (idx // tl.int32(spatial_size)) % tl.int32(num_channels)

    mean_val = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    var_val = tl.load(var_ptr + channel, mask=mask, other=0.0)
    gamma_val = tl.load(gamma_ptr + channel, mask=mask, other=0.0)
    beta_val = tl.load(beta_ptr + channel, mask=mask, other=0.0)

    norm = (input_val - mean_val) / tl.math.sqrt(var_val + 1e-5)
    output_val = norm * gamma_val + beta_val

    tl.store(output_ptr + idx, output_val, mask=mask)

def triton_kernel(
    input,
    output,
    mean,
    var,
    gamma,
    beta,
    batch_size,
    num_channels,
    spatial_size
):
    total = batch_size * num_channels * spatial_size
    block_size = 256
    grid = lambda meta: (total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
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
        BLOCK_SIZE=block_size
    )