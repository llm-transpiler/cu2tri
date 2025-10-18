import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input_ptr, output_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    batch_size, num_channels, spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * num_channels * spatial_size
    mask = idx < total

    # Compute channel index for each element
    channel = (idx // tl.int32(spatial_size)) % tl.int32(num_channels)

    # Load data
    inp = tl.load(input_ptr + idx, mask=mask)
    mean_val = tl.load(mean_ptr + channel, mask=mask)
    var_val = tl.load(var_ptr + channel, mask=mask)
    gamma_val = tl.load(gamma_ptr + channel, mask=mask)
    beta_val = tl.load(beta_ptr + channel, mask=mask)

    # Batch normalization
    denom = tl.math.sqrt(var_val + 1e-5)
    norm = (inp - mean_val) / denom
    out = norm * gamma_val + beta_val

    tl.store(output_ptr + idx, out, mask=mask)

def triton_kernel(
    input, output, mean, var, gamma, beta,
    batch_size, num_channels, spatial_size
):
    total = batch_size * num_channels * spatial_size
    BLOCK_SIZE = 256
    grid = lambda meta: (total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
    _triton_kernel_impl[grid](
        input, output, mean, var, gamma, beta,
        batch_size, num_channels, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )