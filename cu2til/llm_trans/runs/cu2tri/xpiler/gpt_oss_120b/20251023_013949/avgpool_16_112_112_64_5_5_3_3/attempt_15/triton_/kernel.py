import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: 5×5 average pooling (NHWC layout)
# ------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Implements the CUDA kernel `_cuda_kernel_impl` in Triton.
    The kernel assumes NHWC memory layout:
        A      : (batch, height, width, channels)
        pool_avg: (batch, out_height, out_width, channels)
    It computes a 5×5 average pool with stride 3.
    """
    pid = tl.program_id(0)                # blockIdx.x
    t = tl.arange(0, BLOCK_SIZE)          # threadIdx.x (int32 vector)

    # ------------------------------------------------------------------
    # Decompose block and thread indices to recover (b, y_out, x_out, c)
    # ------------------------------------------------------------------
    b = pid
    b_div_81 = b // 81                     # batch index
    b_mod_81 = b % 81

    t_shift8 = t >> 8                      # bits 8‑9 of threadIdx.x
    t_shift6 = t >> 6                      # bits 6‑9 of threadIdx.x
    t_low6   = t & 63                      # bits 0‑5 of threadIdx.x (channel)

    # Constants for the specific problem dimensions:
    #   channels = 64, input_H = input_W = 112, stride = 3
    #   802816 = C*H*W, 21504 = C*H*stride, 7168 = C*W, 192 = C*stride, 64 = C
    term1 = b_div_81 * 802816                              # batch offset
    term2 = ((b_mod_81 * 4 + t_shift8) // 9) * 21504       # y_out * (C*H*stride)
    term3 = ((b * 16 + t_shift6) % 36) * 192               # x_out * (C*stride)

    base_offset = term1 + term2 + term3 + t_low6           # start of 5×5 window

    # ------------------------------------------------------------------
    # Accumulate the sum over the 5×5 region
    # ------------------------------------------------------------------
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        for rv1 in range(5):
            idx = base_offset + rv0 * 7168 + rv1 * 64      # offset for (rv0, rv1)
            idx_i64 = tl.cast(idx, tl.int64)               # pointer arithmetic expects int64
            sum_val += tl.load(A_ptr + idx_i64)

    # ------------------------------------------------------------------
    # Write the average (multiply by 1/25 = 0.04) to the output tensor
    # ------------------------------------------------------------------
    out_offset = pid * BLOCK_SIZE + t                      # linear NHWC index
    out_offset_i64 = tl.cast(out_offset, tl.int64)
    tl.store(pool_avg_ptr + out_offset_i64, sum_val * 0.04)


# ------------------------------------------------------------------
# Wrapper: converts NCHW tensors to NHWC, launches the Triton kernel,
# and converts the result back to NCHW.
# ------------------------------------------------------------------
def triton_kernel(input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Entry‑point that matches the original CUDA kernel signature:
        cuda_kernel(float *input, float *output,
                    int batch_size, int channels,
                    int input_H, int kernel_size, int stride)

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input in NCHW layout (float32, CUDA).
    output_tensor : torch.Tensor
        Output buffer in NCHW layout (float32, CUDA). Will be filled in‑place.
    batch_size, channels, input_H, kernel_size, stride : int
        Dimensions of the pooling operation (must match the kernel constants).
    """
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        raise RuntimeError("input and output tensors must be CUDA tensors")
    if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
        raise RuntimeError("only float32 tensors are supported")

    # Ensure contiguous memory
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not output_tensor.is_contiguous():
        output_tensor = output_tensor.contiguous()

    # ------------------------------------------------------------------
    # Convert NCHW -> NHWC for the kernel
    # ------------------------------------------------------------------
    input_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()

    # Output dimensions
    output_H = (input_H - kernel_size) // stride + 1
    output_W = output_H  # square output in this benchmark
    output_nhwc = torch.empty((batch_size, output_H, output_W, channels),
                              dtype=input_tensor.dtype,
                              device=input_tensor.device)

    # ------------------------------------------------------------------
    # Launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024
    output_size = batch_size * output_H * output_W * channels
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[grid](input_nhwc, output_nhw, BLOCK_SIZE=BLOCK_SIZE)

    # Optional synchronization (useful if subsequent CPU code depends on the result)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Convert NHWC -> NCHW and copy back to the provided output tensor
    # ------------------------------------------------------------------
    output_tensor.copy_(output_nhwc.permute(0, 3, 1, 2))