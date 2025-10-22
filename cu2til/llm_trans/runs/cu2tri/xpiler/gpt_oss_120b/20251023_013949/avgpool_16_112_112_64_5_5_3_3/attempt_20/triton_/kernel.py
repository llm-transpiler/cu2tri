import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, pool_avg,
                        N: tl.constexpr,
                        C: tl.constexpr,
                        H_in: tl.constexpr,
                        K: tl.constexpr,
                        S: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                     # block index
    offs = tl.arange(0, BLOCK_SIZE)            # thread offsets within block

    # Global linear index for the output tensor (flattened NHWC)
    linear_idx = pid * BLOCK_SIZE + offs

    # Output spatial dimensions (square)
    H_out = (H_in - K) // S + 1
    W_out = H_out

    total_output = N * H_out * W_out * C

    # Mask for valid output elements
    mask = linear_idx < total_output

    # Decompose linear index into (n, h_out, w_out, c)
    c = linear_idx % C
    tmp = linear_idx // C
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    n = tmp // H_out

    # Starting coordinates in the input tensor
    h_start = h_out * S
    w_start = w_out * S

    # Base offset for each thread into the input tensor (NHWC layout)
    # index = ((n * H_in + h_start) * H_in + w_start) * C + c
    base = ((n * H_in + h_start) * H_in + w_start) * C + c

    # Offsets for the K×K pooling window
    row_stride = H_in * C          # stride to next input row
    col_stride = C                 # stride to next input column
    idx = tl.arange(0, K * K)      # linear index inside the window
    row_idx = idx // K
    col_idx = idx % K
    offsets = row_idx * row_stride + col_idx * col_stride  # shape [K*K]

    # Final offsets for each thread (shape [BLOCK_SIZE, K*K])
    final_offsets = base[:, None] + offsets[None, :]

    # Load all K×K values (masked)
    vals = tl.load(A + final_offsets, mask=mask[:, None], other=0.0)

    # Sum across the K×K dimension
    sum_val = tl.sum(vals, axis=1)

    # Compute average
    avg = sum_val * (1.0 / (K * K))

    # Write result to output (masked)
    tl.store(pool_avg + linear_idx, avg, mask=mask)


def triton_kernel(input: torch.Tensor,
                  output: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Triton implementation of the average pooling kernel.
    Arguments match the original CUDA kernel signature.
    """
    assert input.is_cuda and output.is_cuda, "Input and output must be CUDA tensors"
    assert input.dtype == torch.float32 and output.dtype == torch.float32, "Only float32 tensors are supported"

    # Compute output spatial dimension
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        input,
        output,
        N=batch_size,
        C=channels,
        H_in=input_H,
        K=kernel_size,
        S=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )