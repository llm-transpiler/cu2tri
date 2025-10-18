import triton
import triton.language as tl
import torch

@triton.jit
def _triton_kernel_impl(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    var_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    batch_size: tl.int32,
    num_channels: tl.int32,
    spatial_size: tl.int32,
    total: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < total

    inp = tl.load(input_ptr + indices, mask=mask)

    # Compute channel index for each element
    channel = (indices // spatial_size) % num_channels

    mean = tl.load(mean_ptr + channel, mask=mask)
    var = tl.load(var_ptr + channel, mask=mask)
    gamma = tl.load(gamma_ptr + channel, mask=mask)
    beta = tl.load(beta_ptr + channel, mask=mask)

    eps = 1e-5
    norm = (inp - mean) / tl.sqrt(var + eps)
    out = norm * gamma + beta

    tl.store(output_ptr + indices, out, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch_size: int,
    num_channels: int,
    spatial_size: int,
):
    """
    Wrapper that launches the Triton kernel with the same interface as the original CUDA kernel.
    All tensors must be on the same CUDA device.
    """
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
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )