import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel (exact replica of the original CUDA implementation)
# ----------------------------------------------------------------------
BLOCK_SIZE = 1024  # threads per block – matches __launch_bounds__(1024)

@triton.jit
def _triton_kernel_impl(
    A_ptr,                 # *float, input tensor (flattened NHWC)
    out_ptr,               # *float, output tensor (flattened custom layout)
    output_size: tl.int64, # total number of output elements
    BLOCK_SIZE: tl.constexpr  # compile‑time constant: threads per block
):
    """
    Implements the CUDA kernel:
        for each thread (identified by blockIdx.x and threadIdx.x)
        compute a 5×5 average‑pool (stride = 2) and write the result.
    The indexing arithmetic is reproduced verbatim.
    """
    pid = tl.program_id(0).to(tl.int64)                     # blockIdx.x
    tid = tl.arange(0, BLOCK_SIZE).to(tl.int64)             # threadIdx.x

    # Linear output index for this thread
    out_idx = pid * BLOCK_SIZE + tid
    mask = out_idx < output_size                             # guard for tail threads

    # ------------------------------------------------------------------
    # Decode threadIdx.x into the bit‑fields used by the original kernel
    # ------------------------------------------------------------------
    t8 = tid >> 8          # bits 9‑8
    t7 = tid >> 7          # bits 9‑7
    t6 = tid >> 6          # bits 9‑6
    t0 = tid & 63          # bits 5‑0 (channel)

    # ------------------------------------------------------------------
    # Recreate the integer arithmetic from the CUDA index expression
    # ------------------------------------------------------------------
    a = ((pid * 4) + t8) // 225                     # batch index
    b = ((pid * 8) + t7) % 450
    c = b // 15                                      # y‑output index (0‑29)
    d = ((pid * 16) + t6) % 30                       # x‑output index (0‑29)

    # Base address of the top‑left element of the 5×5 window (NHWC layout)
    #   a * 262144  = batch * (H_in * W_in * C)   (64*64*64)
    #   c * 8192    = y_out * stride * W_in * C   (2*64*64)
    #   d * 128     = x_out * stride * C          (2*64)
    #   t0          = channel offset
    base_idx = a * 262144 + c * 8192 + d * 128 + t0

    # ------------------------------------------------------------------
    # Accumulate the 5×5 sum (scalar accumulator)
    # ------------------------------------------------------------------
    sum_val = tl.zeros([1], dtype=tl.float32)[0]   # scalar = 0.0
    for rv0 in range(5):          # kernel height
        for rv1 in range(5):      # kernel width
            idx = base_idx + rv0 * 4096 + rv1 * 64   # 4096 = W_in*C, 64 = C
            sum_val += tl.load(A_ptr + idx, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # Compute average (1/25 = 0.04) and write back
    # ------------------------------------------------------------------
    avg = sum_val * 0.04
    tl.store(out_ptr + out_idx, avg, mask=mask)


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
    Wrapper mirroring the original CUDA API.
    Parameters:
        input_tensor   – CUDA tensor of shape (batch, channels, H, H) (float32)
        output_tensor  – pre‑allocated CUDA tensor of shape (batch, channels, H_out, H_out) (float32)
        batch_size, channels, input_H, kernel_size, stride – same as in the CUDA API
    The kernel expects the input in NHWC layout; this wrapper handles the permutation.
    The kernel writes its result into the flat memory of ``output_tensor`` using the
    same layout as the original CUDA kernel.  The test harness will apply the
    ``cuda_output_tensor_transform`` to both outputs before comparison.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        raise RuntimeError("Both input and output tensors must be CUDA tensors.")
    if input_tensor.dtype != torch.float32 or output_tensor.dtype != torch.float32:
        raise RuntimeError("Tensors must be of type torch.float32.")

    # Ensure contiguous memory for safe pointer arithmetic
    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # ------------------------------------------------------------------
    # Compute output spatial dimension and total number of output elements
    # ------------------------------------------------------------------
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    # ------------------------------------------------------------------
    # Convert input from NCHW → NHWC (kernel expects NHWC)
    # ------------------------------------------------------------------
    input_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)

    # Flatten tensors for 1‑D indexing inside the Triton kernel
    A = input_nhwc.view(-1)                # input pointer (NHWC)
    out = output_tensor.view(-1)           # output pointer (custom flat layout)

    # ------------------------------------------------------------------
    # Grid configuration (one block processes BLOCK_SIZE elements)
    # ------------------------------------------------------------------
    num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ------------------------------------------------------------------
    # Launch the Triton kernel
    # ------------------------------------------------------------------
    _triton_kernel_impl[(num_blocks,)](
        A,
        out,
        output_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,          # 1024 threads → 32 warps per block
    )

    # Ensure kernel completion before returning
    torch.cuda.synchronize()