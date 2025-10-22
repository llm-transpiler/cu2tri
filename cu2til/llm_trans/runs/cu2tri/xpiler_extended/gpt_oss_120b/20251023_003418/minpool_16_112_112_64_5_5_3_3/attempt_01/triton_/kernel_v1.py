import torch
import triton
import triton.language as tl

# Triton kernel that reproduces the original CUDA kernel's logic.
@triton.jit
def _triton_kernel_impl(A_ptr, pool_min_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    # blockIdx.x
    pid = tl.program_id(0)
    # threadIdx.x
    lane_id = tl.arange(0, BLOCK_SIZE)

    # Global linear index for the output buffer
    out_idx = pid * BLOCK_SIZE + lane_id
    mask = out_idx < output_size

    # Alias for readability (mirrors CUDA variable names)
    b = pid
    t = lane_id

    # Compute the components of the input index exactly as in the CUDA kernel
    b_div81 = b // 81
    b_mod81 = b % 81

    t_shift8 = t >> 8
    t_shift6 = t >> 6
    t_and63 = t & 63

    # ((blockIdx.x / 81) * 802816)
    part0 = b_div81 * 802816
    # ((((blockIdx.x % 81) * 4 + (threadIdx.x >> 8)) / 9) * 21504)
    part1 = ((b_mod81 * 4 + t_shift8) // 9) * 21504
    # ((((blockIdx.x * 16) + (threadIdx.x >> 6)) % 36) * 192)
    part2 = (((b * 16) + t_shift6) % 36) * 192

    # Base address for the 5×5 window (without rv0/rv1 offsets)
    base = part0 + part1 + part2 + t_and63

    # Initialise reduction with +inf (largest float32)
    min_val = tl.full((BLOCK_SIZE,), 3.402823e+38, dtype=tl.float32)

    # Unrolled 5×5 reduction
    for rv0 in range(5):
        offset_rv0 = rv0 * 7168  # rv0 * 7168
        for rv1 in range(5):
            offset_rv1 = rv1 * 64   # rv1 * 64
            idx = base + offset_rv0 + offset_rv1
            a = tl.load(A_ptr + idx, mask=mask, other=3.402823e+38)
            min_val = tl.minimum(min_val, a)

    # Write the result back to the output buffer
    tl.store(pool_min_ptr + out_idx, min_val, mask=mask)


def triton_kernel(input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int) -> None:
    """
    Triton entry point mirroring the original CUDA kernel signature.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input feature map (float32) residing on a CUDA device.
    output_tensor : torch.Tensor
        Output buffer (float32) residing on a CUDA device.
    batch_size : int
        Number of batches.
    channels : int
        Number of channels.
    input_H : int
        Height (and width) of the square input tensor.
    kernel_size : int
        Size of the pooling kernel (assumed square).
    stride : int
        Stride of the pooling operation.
    """
    # Basic sanity checks
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        raise RuntimeError("Both input and output tensors must reside on a CUDA device.")
    if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
        raise RuntimeError("Both input and output tensors must be of type torch.float32.")

    # Ensure contiguous layout for pointer arithmetic
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Compute output spatial dimension and total output size
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Triton launch configuration (mirrors CUDA's 1024‑thread blocks)
    BLOCK_SIZE = 1024
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,   # 32 warps × 32 threads = 1024 threads per block
    )