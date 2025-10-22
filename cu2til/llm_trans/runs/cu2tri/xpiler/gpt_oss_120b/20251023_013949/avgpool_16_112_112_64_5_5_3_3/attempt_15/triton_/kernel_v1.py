import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A_ptr, pool_avg_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Triton implementation of the 5x5 average pooling kernel.
    Mirrors the original CUDA kernel _cuda_kernel_impl.
    """
    pid = tl.program_id(0)                # blockIdx.x
    t = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)   # threadIdx.x

    # Decompose block and thread indices to match the original indexing scheme
    b = pid
    b_div_81 = b // 81
    b_mod_81 = b % 81

    t_shift8 = t >> 8          # bits 8‑9 of threadIdx.x
    t_shift6 = t >> 6          # bits 6‑9 of threadIdx.x
    t_low6   = t & 63          # bits 0‑5 of threadIdx.x

    # Constants derived from the original CUDA kernel
    term1 = b_div_81 * 802816
    term2 = ((b_mod_81 * 4 + t_shift8) // 9) * 21504
    term3 = ((b * 16 + t_shift6) % 36) * 192

    # Base offset for the 5×5 window
    base_offset = term1 + term2 + term3 + t_low6

    # Accumulate the sum over the 5×5 region
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for rv0 in range(5):
        for rv1 in range(5):
            idx = base_offset + rv0 * 7168 + rv1 * 64
            sum_val += tl.load(A_ptr + idx)

    # Write the average (multiply by 1/25 = 0.04)
    out_ptr = pool_avg_ptr + pid * BLOCK_SIZE + t
    tl.store(out_ptr, sum_val * 0.04)


def triton_kernel(input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Wrapper that launches the Triton kernel.
    The signature matches the original CUDA kernel:
    cuda_kernel(float *input, float *output, int batch_size,
                int channels, int input_H, int kernel_size, int stride)
    """
    # Basic sanity checks
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        raise RuntimeError("input and output tensors must be CUDA tensors")
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not output_tensor.is_contiguous():
        output_tensor = output_tensor.contiguous()

    # Compute output dimensions (identical to the CUDA host code)
    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](input_tensor, output_tensor, BLOCK_SIZE=BLOCK_SIZE)
    # Optional: synchronize if subsequent CPU code depends on the result
    # torch.cuda.synchronize()