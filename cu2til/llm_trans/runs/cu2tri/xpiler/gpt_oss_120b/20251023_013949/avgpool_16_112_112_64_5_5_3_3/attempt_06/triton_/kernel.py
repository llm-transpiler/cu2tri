import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, output_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)               # blockIdx.x
    offsets = tl.arange(0, BLOCK_SIZE)    # threadIdx.x

    out_idx = pid * BLOCK_SIZE + offsets
    valid = out_idx < output_size

    # Accumulator for each thread (vectorized across the block)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Pre‑compute parts of the input index that are independent of the 5×5 window
    batch = pid // 81
    block_in_batch = pid % 81
    # batch offset
    i0 = batch * 802816
    # vertical offset: ((block_in_batch * 4 + (offsets >> 8)) // 9) * (stride * height_stride)
    i1 = ((block_in_batch * 4 + (offsets >> 8)) // 9) * 21504
    # horizontal offset: ((pid * 16 + (offsets >> 6)) % 36) * (stride * width_stride)
    i3 = ((pid * 16 + (offsets >> 6)) % 36) * 192
    # channel offset
    i5 = offsets & 63

    # Loop over the 5×5 pooling window
    for rv0 in tl.static_range(5):
        i2 = rv0 * 7168          # height_stride * rv0
        for rv1 in tl.static_range(5):
            i4 = rv1 * 64         # width_stride * rv1
            idx = i0 + i1 + i2 + i3 + i4 + i5
            a = tl.load(A_ptr + idx, mask=valid, other=0.0)
            sum_val += a

    # Write the average (sum * 1/25) to the output
    tl.store(pool_avg_ptr + out_idx, sum_val * 0.04, mask=valid)


def triton_kernel(input: torch.Tensor, output: torch.Tensor,
                  batch_size: int, channels: int, input_H: int,
                  kernel_size: int, stride: int):
    """
    Triton implementation of the average‑pooling kernel.
    The signature matches the original CUDA host function:
        cuda_kernel(float *input, float *output,
                    int batch_size, int channels,
                    int input_H, int kernel_size, int stride)

    The original CUDA kernel assumes NHWC layout (batch, H, W, C) with a
    fixed 5×5 kernel and stride = 3.  This wrapper permutes the user‑provided
    NCHW tensors to NHWC, launches the kernel, and permutes the result back.
    """
    if not input.is_cuda or not output.is_cuda:
        raise RuntimeError("input and output must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("input and output must be torch.float32")

    # Convert NCHW → NHWC
    input_nhwc = input.permute(0, 2, 3, 1).contiguous()

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_nhwc = torch.empty(batch_size, output_H, output_H, channels,
                              dtype=torch.float32, device=input.device)

    output_size = batch_size * output_H * output_H * channels
    BLOCK_SIZE = 1024
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[(num_blocks,)](
        input_nhwc,
        output_nhwc,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

    # Convert NHWC → NCHW and copy into the user‑provided output tensor
    output.copy_(output_nhwc.permute(0, 3, 1, 2).contiguous())