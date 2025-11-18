import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel_impl(
    A,                     # *float32 input (NHWC)
    pool_avg,              # *float32 output (NHWC)
    N: tl.constexpr,       # batch size
    C: tl.constexpr,       # channels
    H_in: tl.constexpr,    # input height (square)
    K: tl.constexpr,       # kernel size
    S: tl.constexpr,       # stride
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                     # block index (== blockIdx.x)
    offs = tl.arange(0, BLOCK_SIZE)            # thread offsets within block (== threadIdx.x)

    # Global linear index of the output element (flattened NHWC)
    linear_idx = pid * BLOCK_SIZE + offs

    # Output spatial dimensions (square)
    H_out = (H_in - K) // S + 1
    W_out = H_out
    total_output = N * H_out * W_out * C

    # Mask for threads that correspond to a valid output element
    mask = linear_idx < total_output

    # Decompose linear index into (n, h_out, w_out, c) for NHWC layout
    c = linear_idx % C
    tmp = linear_idx // C
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    n = tmp // H_out

    # Top‑left corner of the K×K window in the input tensor
    h_start = h_out * S
    w_start = w_out * S

    # Base offset for each thread (NHWC layout)
    # offset = ((n * H_in + h_start) * H_in + w_start) * C + c
    base = ((n * H_in + h_start) * H_in + w_start) * C + c

    # Strides for moving inside the input tensor (NHWC)
    row_stride = H_in * C   # one row down (H_in * C)
    col_stride = C          # one column right (C)

    # Accumulator for the K×K window (one scalar per thread)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the pooling window
    for i in range(K):
        for j in range(K):
            offset = base + i * row_stride + j * col_stride
            # Masked load: out‑of‑range threads read 0.0
            val = tl.load(A + offset, mask=mask, other=0.0)
            sum_val = sum_val + val

    # Compute average
    avg = sum_val * (1.0 / (K * K))

    # Write result (masked)
    tl.store(pool_avg + linear_idx, avg, mask=mask)


def triton_kernel(
    input: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    channels: int,
    input_H: int,
    kernel_size: int,
    stride: int,
):
    """
    Triton implementation of the average‑pooling kernel.
    The signature matches the original CUDA kernel.
    The kernel expects NHWC layout for both input and output.
    If tensors are provided in NCHW layout, they permuted internally.
    """
    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    if not (input.is_cuda and output.is_cuda):
        raise RuntimeError("Input and output tensors must be CUDA tensors")
    if input.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Only float32 tensors are supported")

    # ----------------------------------------------------------------------
    # Determine layout (NHWC vs NCHW) and prepare contiguous tensors
    # ----------------------------------------------------------------------
    # Input layout detection
    if input.dim() != 4:
        raise RuntimeError("Input tensor must be 4‑dimensional")
    if input.shape[1] == channels and input.shape[2] == input_H and input.shape[3] == input_H:
        # NCHW layout
        input_nhwc = input.permute(0, 2, 3, 1).contiguous()
        input_is_nchw = True
    elif input.shape[-1] == channels and input.shape[1] == input_H and input.shape[2] == input_H:
        # NHWC layout
        input_nhwc = input if input.is_contiguous() else input.contiguous()
        input_is_nchw = False
    else:
        raise RuntimeError("Unable to infer input layout (expected NHWC or NCHW)")

    # Output layout detection
    output_H = (input_H - kernel_size) // stride + 1
    if output.dim() != 4:
        raise RuntimeError("Output tensor must be 4‑dimensional")
    if output.shape[1] == channels and output.shape[2] == output_H and output.shape[3] == output_H:
        # NCHW layout
        output_is_nchw = True
        # Allocate a temporary NHWC buffer for the kernel
        output_nhwc = torch.empty(
            batch_size, output_H, output_H, channels,
            device=output.device,
            dtype=output.dtype,
        )
    elif output.shape[-1] == channels and output.shape[1] == output_H and output.shape[2] == output_H:
        # NHWC layout
        output_is_nchw = False
        output_nhwc = output if output.is_contiguous() else output.contiguous()
    else:
        raise RuntimeError("Unable to infer output layout (expected NHWC or NCHW)")

    # ----------------------------------------------------------------------
    # Launch configuration
    # ----------------------------------------------------------------------
    output_size = batch_size * output_H * output_H * channels
    BLOCK_SIZE = 1024                     # must be a power of 2 for tl.arange
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_tuple = (grid,)

    # ----------------------------------------------------------------------
    # Kernel launch
    # ----------------------------------------------------------------------
    _triton_kernel_impl[grid_tuple](
        input_nhwc,
        output_nhwc,
        N=batch_size,
        C=channels,
        H_in=input_H,
        K=kernel_size,
        S=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

    # ----------------------------------------------------------------------
    # Convert output back to the original layout if needed
    # ----------------------------------------------------------------------
    if output_is_nchw:
        # NHWC -> NCHW
        output.copy_(output_nhwc.permute(0, 3, 1, 2).contiguous())
    else:
        # Already NHWC; ensure the original tensor is updated
        if not output.is_contiguous():
            output.copy_(output_nhwc)