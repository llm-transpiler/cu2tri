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
    spatial_size: tl.constexpr,
    num_channels: tl.constexpr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < total

    inp = tl.load(input_ptr + indices, mask=mask)
    # channel index for each element
    channel = tl.math.floor_divide(indices, spatial_size) % num_channels

    mean = tl.load(mean_ptr + channel, mask=mask)
    var = tl.load(var_ptr + channel, mask=mask)
    gamma = tl.load(gamma_ptr + channel, mask=mask)
    beta = tl.load(beta_ptr + channel, mask=mask)

    norm = (inp - mean) / tl.math.sqrt(var + 1e-5)
    out = norm * gamma + beta

    tl.store(output_ptr + indices, out, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
):
    """
    Wrapper that matches the original CUDA kernel signature.
    Expects input, output, mean, var, gamma, beta tensors.
    """
    # Ensure tensors are on the same device
    device = input.device
    mean = mean.to(device)
    var = var.to(device)
    gamma = gamma.to(device)
    beta = beta.to(device)

    # Compute dimensions
    if input.ndim == 3:
        batch_size, num_channels, spatial_size = input.shape
    else:
        # Fallback for 1-D flattened input
        batch_size = 1
        num_channels = mean.shape[0]
        spatial_size = input.shape[0] // num_channels

    total = batch_size * num_channels * spatial_size

    # Flatten input and output for the kernel
    input_flat = input.reshape(-1)
    output_flat = output.reshape(-1)

    BLOCK_SIZE = 256
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    _triton_kernel_impl[(grid,)](
        input_flat,
        output_flat,
        mean,
        var,
        gamma,
        beta,
        spatial_size,
        num_channels,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output