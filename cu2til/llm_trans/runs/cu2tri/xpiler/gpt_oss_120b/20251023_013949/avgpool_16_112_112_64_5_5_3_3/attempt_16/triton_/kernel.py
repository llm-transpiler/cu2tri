import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: average pooling with a 5×5 window and stride 3.
# Layout: blocked NCHWc with channel block size = 64.
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                     # *float32  (input)
    B_ptr,                     # *float32  (output)
    batch_size: tl.int32,
    channels: tl.int32,
    input_H: tl.int32,
    kernel_size: tl.constexpr, # compile‑time constant (5 in the original CUDA)
    stride: tl.constexpr,      # compile‑time constant (3 in the original CUDA)
    total_output: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # linear thread ids
    mask = offsets < total_output

    # ------------------------------------------------------------------
    # Output spatial dimension (same formula as the host code)
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1

    # ------------------------------------------------------------------
    # Decode NCHWc coordinates from the linear output offset.
    # ------------------------------------------------------------------
    c_inner = offsets % 64                                 # inner channel (0..63)
    tmp = offsets // 64
    out_x = tmp % output_H                                 # output column
    tmp = tmp // output_H
    out_y = tmp % output_H                                 # output row
    tmp = tmp // output_H
    c_block = tmp % (channels // 64)                       # channel block index
    batch = tmp // (channels // 64)                         # batch index

    # ------------------------------------------------------------------
    # Compute the top‑left corner of the 5×5 window in the input tensor.
    # ------------------------------------------------------------------
    in_y = out_y * stride
    in_x = out_x * stride

    # Base offset for the first element of the window:
    # ((batch * (C//64) + c_block) * H + in_y) * H + in_x) * 64 + c_inner
    base = ((batch * (channels // 64) + c_block) * input_H + in_y) * input_H + in_x
    base = base * 64 + c_inner

    # ------------------------------------------------------------------
    # Accumulate the sum over the 5×5 region.
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for ky in tl.static_range(kernel_size):
        for kx in tl.static_range(kernel_size):
            idx = base + ky * input_H * 64 + kx * 64
            sum_val += tl.load(A_ptr + idx, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # Compute the average (1/25 = 0.04) and write back.
    # ------------------------------------------------------------------
    avg = sum_val * (1.0 / (kernel_size * kernel_size))
    tl.store(B_ptr + offsets, avg, mask=mask)


# ----------------------------------------------------------------------
# Python wrapper that mimics the original CUDA launch interface.
# ----------------------------------------------------------------------
def triton_kernel(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Entry point with the same signature as the original `cuda_kernel`.
    The tensors must be in the blocked NCHWc layout (channel block = 64).
    """
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be CUDA tensors"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32, "Only float32 is supported"
    assert channels % 64 == 0, "channels must be a multiple of 64 for the blocked layout"

    # Output spatial dimension (same as host code)
    output_H = (input_H - kernel_size) // stride + 1
    total_output = batch_size * channels * output_H * output_H

    BLOCK_SIZE = 1024
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        batch_size,
        channels,
        input_H,
        kernel_size,   # constexpr
        stride,        # constexpr
        total_output,
        BLOCK_SIZE,
        num_warps=32,  # 1024 threads = 32 warps
    )