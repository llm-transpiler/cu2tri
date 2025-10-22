import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(A, pool_min, output_size, input_size,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE, dtype=tl.int64)
    b = tl.int64(pid)
    t = offs

    # Global linear thread id
    tid = b * BLOCK_SIZE + t
    mask = tid < output_size

    # Decompose block and thread indices as in the original CUDA kernel
    b_div_81 = b // 81
    b_mod_81 = b % 81
    t_shift8 = t >> 8
    t_shift6 = t >> 6
    t_and63 = t & 63

    term1 = b_div_81 * 802816
    term2 = ((b_mod_81 * 4 + t_shift8) // 9) * 21504
    term4 = (((b * 16 + t_shift6) % 36) * 192)
    term6 = t_and63

    base = term1 + term2 + term4 + term6

    # Initialize per‑thread minimum with the largest float32 value
    max_float = 3.402823e+38
    min_val = tl.full([BLOCK_SIZE], max_float, dtype=tl.float32)

    # 5×5 window reduction
    for rv0 in range(5):
        offset_rv0 = rv0 * 7168
        for rv1 in range(5):
            idx = base + offset_rv0 + rv1 * 64
            load_mask = mask & (idx < input_size)
            val = tl.load(A + idx, mask=load_mask, other=max_float)
            min_val = tl.minimum(min_val, val)

    # Write the result back
    tl.store(pool_min + tid, min_val, mask=mask)


def triton_kernel(input_tensor: torch.Tensor,
                  output_tensor: torch.Tensor,
                  batch_size: int,
                  channels: int,
                  input_H: int,
                  kernel_size: int,
                  stride: int):
    """
    Triton entry point mirroring the original `cuda_kernel` signature.
    `input_tensor` and `output_tensor` must be 1‑D CUDA tensors of dtype float32.
    """
    assert input_tensor.is_cuda and output_tensor.is_cuda, "Tensors must be on CUDA"
    assert input_tensor.dtype == torch.float32 and output_tensor.dtype == torch.float32

    output_H = (input_H - kernel_size) // stride + 1
    output_size = batch_size * output_H * output_H * channels
    input_size = batch_size * channels * input_H * input_H

    BLOCK_SIZE = 1024
    grid = ((output_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _triton_kernel_impl[grid](
        input_tensor,
        output_tensor,
        output_size,
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    # Example usage
    batch = 2
    channels = 3
    H = 112
    kernel = 5
    stride = 1

    out_H = (H - kernel) // stride + 1
    # Allocate tensors in flattened layout
    x = torch.randn(batch * channels * H * H, device="cuda", dtype=torch.float32)
    y = torch.empty(batch * channels * out_H * out_H, device="cuda", dtype=torch.float32)

    triton_kernel(x, y, batch, channels, H, kernel, stride)

    # Print a few output values for sanity check
    print("First 10 output values:", y[:10].cpu().numpy())