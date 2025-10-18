import torch
import triton
import triton.language as tl

# Block size for each kernel launch
BLOCK_SIZE = 256

@triton.jit
def _triton_kernel_impl(
    input_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size: tl.int32,
    num_channels: tl.int32,
    spatial_size: tl.int32,
    BLOCK_SIZE: tl.int32
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * num_channels * spatial_size
    mask = offset < total

    idx = offset
    channel = (idx // spatial_size) % num_channels

    inp = tl.load(input_ptr + idx, mask=mask, other=0.0)
    mean_val = tl.load(mean_ptr + channel)
    var_val = tl.load(var_ptr + channel)
    gamma_val = tl.load(gamma_ptr + channel)
    beta_val = tl.load(beta_ptr + channel)

    norm = (inp - mean_val) / tl.math.sqrt(var_val + 1e-5)
    out = norm * gamma_val + beta_val

    tl.store(output_ptr + idx, out, mask=mask)

def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    batch_size: int,
    num_channels: int,
    spatial_size: int
):
    """
    Triton wrapper that mimics the CUDA kernel signature.
    All tensors must be on the same CUDA device and of dtype torch.float32.
    """
    total = batch_size * num_channels * spatial_size
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_kernel_impl[grid, BLOCK_SIZE](
        input,
        output,
        mean,
        var,
        gamma,
        beta,
        batch_size,
        num_channels,
        spatial_size,
        BLOCK_SIZE
    )