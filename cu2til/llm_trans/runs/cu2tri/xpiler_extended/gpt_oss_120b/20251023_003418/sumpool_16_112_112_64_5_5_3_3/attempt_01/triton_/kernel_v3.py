import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that reproduces the original CUDA implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    """
    Compute a 5×5 sum (average‑pooling numerator) for each output element.
    The indexing logic mirrors the CUDA kernel exactly.
    """
    # Program (block) identifier corresponds to blockIdx.x
    pid = tl.program_id(0)

    # Thread indices within the block (threadIdx.x)
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)  # shape: (BLOCK_SIZE,)

    # Linear output index: pool_avg[blockIdx.x * BLOCK_SIZE + threadIdx.x]
    out_idx = pid * BLOCK_SIZE + offsets
    mask = out_idx < output_size  # guard out‑of‑range threads

    # ------------------------------------------------------------------
    # Recreate the address computation from the CUDA kernel
    # ------------------------------------------------------------------
    b = pid
    t = offsets

    # term1 = (blockIdx.x / 81) * 802816
    term1 = (b // 81) * 802816

    # term2 = (( (blockIdx.x % 81) * 4 + (threadIdx.x >> 8) ) / 9) * 21504
    temp = (b % 81) * 4 + (t >> 8)
    term2 = (temp // 9) * 21504

    # term4 = (( (blockIdx.x * 16) + (threadIdx.x >> 6) ) % 36) * 192
    term4 = ((b * 16 + (t >> 6)) % 36) * 192

    # term6 = threadIdx.x & 63
    term6 = t & 63

    # Base address (independent of the 5×5 window offsets)
    base = term1 + term2 + term4 + term6  # shape: (BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Offsets for the 5×5 pooling window (rv0 ∈ [0,4], rv1 ∈ [0,4])
    # ------------------------------------------------------------------
    i = tl.arange(0, 25).to(tl.int64)          # enumerate the 25 positions
    rv0 = i // 5
    rv1 = i % 5
    window_offset = rv0 * 7168 + rv1 * 64       # shape: (25,)

    # Absolute indices of the 25 elements for each thread
    # Shape: (25, BLOCK_SIZE)
    idx = base[None, :] + window_offset[:, None]

    # Load the 25 values (masked for threads that are out of the valid output range)
    vals = tl.load(A_ptr + idx, mask=mask[None, :], other=0.0)

    # Sum across the window dimension → shape (BLOCK_SIZE,)
    pool_sum = tl.sum(vals, axis=0)

    # Write the result back to the output tensor
    tl.store(pool_avg_ptr + out_idx, pool_sum, mask=mask)


# ----------------------------------------------------------------------
# Wrapper with the exact signature of the original CUDA host function
# ----------------------------------------------------------------------
def triton_kernel(input: torch.Tensor,
                  output: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Triton entry point that mirrors the original ``cuda_kernel`` signature.
    """
    # Basic sanity checks
    assert input.is_cuda and output.is_cuda, "Tensors must reside on CUDA"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, \
        "Only float32 tensors are supported"

    # Compute output spatial dimension and total number of output elements
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Block size must match the launch bounds of the CUDA kernel
    BLOCK_SIZE = 1024

    # One‑dimensional grid configuration
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Ensure kernel completion before returning
    torch.cuda.synchronize()