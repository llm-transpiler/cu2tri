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
    pid = tl.program_id(axis=0)
    stride = BLOCK_SIZE
    start = pid * stride
    end = start + stride
    total = batch_size * num_channels * spatial_size

    for i in range(start, end):
        if i < total:
            channel = (i // spatial_size) % num_channels
            val = tl.load(input_ptr + i)
            m = tl.load(mean_ptr + channel)
            v = tl.load(var_ptr + channel)
            g = tl.load(gamma_ptr + channel)
            b = tl.load(beta_ptr + channel)
            norm = (val - m) / tl.sqrt(v + 1e-5)
            tl.store(output_ptr + i, norm * g + b)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor
) -> torch.Tensor:
    """
    Triton implementation of the batchnorm kernel.
    Parameters are expected to be 1‑D tensors for mean, var, gamma, beta
    and a 4‑D input/output tensor of shape (N, C, H, W).
    """
    batch_size = input.shape[0]
    num_channels = mean.shape[0]
    spatial_size = input.shape[2] * input.shape[3] if input.ndim >= 4 else 1

    total = batch_size * num_channels * spatial_size
    BLOCK_SIZE = 256
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    _triton_kernel_impl[(grid,)](
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
    return output