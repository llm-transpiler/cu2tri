import torch
import triton
import triton.language as tl

# Number of threads per block (matches __launch_bounds__(1024) in the CUDA kernel)
BLOCK_SIZE = 1024

@triton.jit
def _triton_kernel_impl(
    A,                     # pointer to input tensor (float*)
    pool_avg,              # pointer to output tensor (float*)
    output_size,           # total number of output elements (int)
    BLOCK_SIZE: tl.constexpr  # compile‑time constant: threads per block
):
    """
    Direct translation of the CUDA _cuda_kernel_impl to Triton.
    Computes a 5×5 average‑pooling sum and writes the average.
    """
    # Program (block) identifier
    pid = tl.program_id(0).to(tl.int64)          # blockIdx.x
    # Linear offsets for the output elements handled by this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    mask = offsets < output_size                  # guard for the tail

    # Thread identifier within the block (matches threadIdx.x)
    threadIdx = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)

    # Decompose threadIdx into the bit‑fields used by the original kernel
    t8 = threadIdx >> 8          # bits 9‑8
    t7 = threadIdx >> 7          # bits 9‑7
    t6 = threadIdx >> 6          # bits 9‑6
    t0 = threadIdx & 63          # bits 5‑0

    # Recreate the integer arithmetic from the CUDA index expression
    a = ((pid * 4) + t8) // 225
    b = ((pid * 8) + t7) % 450
    c = b // 15
    d = ((pid * 16) + t6) % 30

    # Base address for the 5×5 window (without the rv0/rv1 offsets)
    base_idx = a * 262144 + c * 8192 + d * 128 + t0

    # Accumulate the 5×5 sum
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):          # unrolled at compile time
        for rv1 in range(5):
            idx = base_idx + rv0 * 4096 + rv1 * 64
            sum_val += tl.load(A + idx, mask=mask, other=0.0)

    # Compute the average (1/25 = 0.04) and write back
    avg = sum_val * 0.04
    tl.store(pool_avg + offsets, avg, mask=mask)


def triton_kernel(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int
):
    """
    Triton entry‑point that mirrors the original CUDA kernel signature.
    Parameters:
        input_tensor  – CUDA tensor of shape (batch, H, H, channels) (float32)
        output_tensor – pre‑allocated CUDA tensor for the result (float32)
        batch_size, channels, input_H, kernel_size, stride – same as in the CUDA API
    The function launches the Triton kernel and synchronises the device.
    """
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        raise RuntimeError("Both input and output tensors must be CUDA tensors.")
    if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
        raise RuntimeError("Tensors must be of type torch.float32.")

    # Ensure contiguous memory layout
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Compute output spatial dimension and total number of output elements
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # Flatten tensors for 1‑D indexing inside the Triton kernel
    A = input_tensor.view(-1)
    pool_avg = output_tensor.view(-1)

    # Grid configuration (one block processes BLOCK_SIZE elements)
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        A,
        pool_avg,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,          # 1024 threads → 32 warps per block
    )
    # Ensure the kernel has finished before returning
    torch.cuda.synchronize()