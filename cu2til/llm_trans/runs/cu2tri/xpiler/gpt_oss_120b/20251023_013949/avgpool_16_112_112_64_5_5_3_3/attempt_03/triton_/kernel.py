import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that mirrors the original CUDA implementation
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    A_ptr,                # float*  (input tensor)
    pool_avg_ptr,         # float*  (output tensor)
    output_size,          # int32   (total number of output elements)
    BLOCK_SIZE: tl.constexpr,  # compile‑time block size (must be 1024)
):
    # ------------------------------------------------------------------
    # Thread and block identifiers (equivalent to CUDA's blockIdx.x and threadIdx.x)
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                              # blockIdx.x
    tid = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)     # threadIdx.x

    # Linear index of the output element processed by this thread
    out_idx = pid * BLOCK_SIZE + tid
    mask = out_idx < output_size                         # guard against out‑of‑bounds threads

    # ------------------------------------------------------------------
    # Hard‑coded constants derived from the original CUDA kernel
    # (these correspond to the specific shape used in the CUDA code:
    #  channels = 16, input_H = input_W = 224)
    # ------------------------------------------------------------------
    const1 = 802816   # channels * input_H * input_W
    const2 = 21504    # input_H * channels * 6
    const3 = 7168     # input_H * channels * 2
    const4 = 192      # channels * 12
    const5 = 64       # tile width

    # ------------------------------------------------------------------
    # Compute the base address in the input tensor for the 5×5 window
    # ------------------------------------------------------------------
    block_div_81 = pid // 81
    block_mod_81 = pid % 81

    part1 = block_div_81 * const1
    part2 = ((block_mod_81 * 4 + (tid >> 8)) // 9) * const2
    part3 = ((pid * 16 + (tid >> 6)) % 36) * const4
    part4 = tid & 63

    base = part1 + part2 + part3 + part4

    # ------------------------------------------------------------------
    # Accumulate the sum over the 5×5 pooling window
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for rv0 in range(5):
        for rv1 in range(5):
            offset = base + rv0 * const3 + rv1 * const5
            a = tl.load(A_ptr + offset, mask=mask, other=0.0)
            sum_val += a

    # ------------------------------------------------------------------
    # Write the average (sum * 0.04) to the output tensor
    # ------------------------------------------------------------------
    avg = sum_val * 0.04
    tl.store(pool_avg_ptr + out_idx, avg, mask=mask)


# ----------------------------------------------------------------------
# Wrapper function matching the original CUDA kernel signature
# ----------------------------------------------------------------------
def triton_kernel(
    input: torch.Tensor,   # float* input tensor
    output: torch.Tensor,  # float* output tensor
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Entry point that mirrors the original CUDA kernel signature.
    Launches the Triton kernel with the same semantics.
    """
    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Ensure tensors are on CUDA and contiguous
    if not input.is_cuda:
        input = input.cuda()
    if not output.is_cuda:
        output = output.cuda()
    input = input.contiguous()
    output = output.contiguous()

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )