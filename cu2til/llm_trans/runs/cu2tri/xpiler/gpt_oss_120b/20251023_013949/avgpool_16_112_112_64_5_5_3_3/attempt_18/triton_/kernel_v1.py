import torch
import triton
import triton.language as tl

def triton_kernel(input: torch.Tensor, output: torch.Tensor,
                 batch_size: int, channels: int,
                 input_H: int, kernel_size: int, stride: int):
    """
    Triton implementation of the average‑pooling kernel.
    Mirrors the original CUDA kernel's behavior and signature.
    """
    # Compute output dimensions
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels
    BLOCK_SIZE = 1024
    inv_kernel_area = 1.0 / (kernel_size * kernel_size)

    @triton.jit
    def _triton_kernel_impl(A, pool_avg):
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pid < output_size

        # Decode linear index into (n, oh, ow, c) assuming NHWC layout
        c = pid % channels
        tmp = pid // channels
        ow = tmp % output_H
        tmp = tmp // output_H
        oh = tmp % output_H
        n = tmp // output_H

        # Top‑left corner of the kernel window in the input tensor
        in_y0 = oh * stride
        in_x0 = ow * stride
        base = ((n * input_H + in_y0) * input_H + in_x0) * channels + c

        # Accumulate the sum over the kernel window
        sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for rv0 in range(kernel_size):
            for rv1 in range(kernel_size):
                offset = base + (rv0 * input_H + rv1) * channels
                sum_val += tl.load(A + offset, mask=mask, other=0.0)

        avg = sum_val * inv_kernel_area
        tl.store(pool_avg + pid, avg, mask=mask)

    # Launch configuration
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _triton_kernel_impl[grid](input, output)