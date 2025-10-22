import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    input1_ptr, input2_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton implementation of the concat kernel.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # linear indices for this program
    total = N * (C * 2) * H * W
    mask = offs < total

    # Compute coordinates from the linear index
    hw = H * W
    cout = C * 2

    n = offs // (cout * hw)
    rem = offs % (cout * hw)
    c = rem // hw
    offset = rem % hw

    # Base offset for the current batch element in the input tensors
    base = n * hw

    # Channel index inside a single input tensor (0 .. C-1)
    c_mod = tl.where(c < C, c, c - C)

    # Linear index inside the input tensors
    in_idx = base + c_mod * hw + offset

    # Masks for the two possible sources
    mask_c1 = (c < C) & mask
    mask_c2 = (c >= C) & mask

    # Load from the appropriate input; the non‑selected load returns 0.0
    val1 = tl.load(input1_ptr + in_idx, mask=mask_c1, other=0.0)
    val2 = tl.load(input2_ptr + in_idx, mask=mask_c2, other=0.0)

    out_val = val1 + val2  # exactly one of val1/val2 is non‑zero

    # Write the result
    tl.store(output_ptr + offs, out_val, mask=mask)


def triton_kernel(input1, input2, output, N, C, H, W):
    """
    Wrapper that mimics the original CUDA kernel signature.

    Parameters
    ----------
    input1 : torch.Tensor
        Tensor of shape (N, C, H, W), dtype torch.float32, CUDA device.
    input2 : torch.Tensor
        Tensor of shape (N, C, H, W), dtype torch.float32, CUDA device.
    output : torch.Tensor
        Tensor of shape (N, 2*C, H, W), dtype torch.float32, CUDA device.
    N, C, H, W : int
        Dimensions of the input tensors.
    """
    # Basic sanity checks
    assert input1.is_cuda and input2.is_cuda and output.is_cuda, "All tensors must reside on CUDA device"
    assert input1.dtype == torch.float32 and input2.dtype == torch.float32 and output.dtype == torch.float32
    assert input1.is_contiguous() and input2.is_contiguous() and output.is_contiguous()

    total = N * (C * 2) * H * W
    BLOCK_SIZE = 128  # can be tuned; 128 works well on H800

    # Grid definition: one program per BLOCK_SIZE elements
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _triton_kernel_impl[grid](
        input1,
        input2,
        output,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )