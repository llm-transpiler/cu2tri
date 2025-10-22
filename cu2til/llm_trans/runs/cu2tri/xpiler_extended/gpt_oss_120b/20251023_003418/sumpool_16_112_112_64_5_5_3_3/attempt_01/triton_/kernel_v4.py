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
    pid = tl.program_id(0)

    # thread indices within the block
    offsets = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)

    out_idx = pid * BLOCK_SIZE + offsets
    mask = out_idx < output_size

    b = pid
    t = offsets

    term1 = (b // 81) * 802816
    temp = (b % 81) * 4 + (t >> 8)
    term2 = (temp // 9) * 21504
 term4 = (( * 16 +t 6)) % 36) * 192
    term6 = t & 63

    base = term1 + term2 + term4 + term6

    # indices for the 5×5 pooling window
    i = tl.arange(0, 32, dtype=tl.int64)  # 32 ≥ 25 and is a power of 2
    mask_i = i < 25
    rv0 = i // 5
    rv1 = i % 5
    window_offset = rv0 * 7168 + rv1 * 64
    window_offset = tl.where(mask_i, window_offset, 0)

    idx = base[None, :] + window_offset[:, None]

    load_mask = mask_i[:, None] & mask[None, :]
    vals = tl.load(A_ptr + idx, mask=load_mask, other=0.0)

    pool_sum = tl.sum(vals, axis=0)

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
    """Entry point matching the original ``cuda_kernel`` signature."""
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 supported"

    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()