import triton
import triton.language as tl
import torch
import numpy as np

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr, CUDA_BLOCK: tl.constexpr):
    """
    Each program processes BLOCK lanes (BLOCK must be a power of two).
    We map the first CUDA_BLOCK lanes to the CUDA-style threads [0..CUDA_BLOCK)
    and mask out the remaining lanes. This reproduces a global index:
      idx = program_id(0) * CUDA_BLOCK + thread_in_block
    where thread_in_block in [0, CUDA_BLOCK).
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)                     # BLOCK must be power-of-two
    offs = pid * CUDA_BLOCK + lane
    mask = (lane < CUDA_BLOCK) & (offs < size)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper that matches the CUDA kernel signature: triton_kernel(A, B, C, size)
    - Performs elementwise addition C[i] = A[i] + B[i] for i in [0, size).
    - Mirrors the intended behavior of a CUDA kernel launched with blockDim=320.
    """
    CUDA_BLOCK = 320
    # Choose a Triton BLOCK that is a power of two and >= CUDA_BLOCK
    TRITON_BLOCK = 1 << ((CUDA_BLOCK - 1).bit_length())  # for 320 -> 512

    size = int(size)
    if size <= 0:
        return

    def _to_torch_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return torch.as_tensor(x)

    # Prepare A
    A_in = _to_torch_tensor(A)
    if not (torch.is_tensor(A_in) and A_in.is_cuda and A_in.dtype == torch.float32 and A_in.is_contiguous()):
        A_gpu = A_in.contiguous().to(device='cuda', dtype=torch.float32)
    else:
        A_gpu = A_in.contiguous()
    A_gpu = A_gpu.view(-1)

    # Prepare B
    B_in = _to_torch_tensor(B)
    if not (torch.is_tensor(B_in) and B_in.is_cuda and B_in.dtype == torch.float32 and B_in.is_contiguous()):
        B_gpu = B_in.contiguous().to(device='cuda', dtype=torch.float32)
    else:
        B_gpu = B_in.contiguous()
    B_gpu = B_gpu.view(-1)

    # Prepare C (may need to copy back)
    C_in = _to_torch_tensor(C)
    orig_C = C
    if torch.is_tensor(C_in) and C_in.is_cuda and C_in.dtype == torch.float32 and C_in.is_contiguous():
        C_gpu = C_in.view(-1)
        copy_back = False
    else:
        C_gpu = C_in.contiguous().to(device='cuda', dtype=torch.float32).view(-1)
        copy_back = True

    # Validate lengths
    if A_gpu.numel() < size or B_gpu.numel() < size or C_gpu.numel() < size:
        raise ValueError(
            f"Input tensors must have at least 'size' elements: "
            f"got A={A_gpu.numel()}, B={B_gpu.numel()}, C={C_gpu.numel()}, size={size}"
        )

    # Launch Triton kernel using the same number of CUDA-like blocks as original code
    num_blocks = (size + CUDA_BLOCK - 1) // CUDA_BLOCK
    if num_blocks > 0:
        _triton_kernel_impl[(num_blocks,)](
            A_gpu, B_gpu, C_gpu, size, BLOCK=TRITON_BLOCK, CUDA_BLOCK=CUDA_BLOCK
        )

    # Copy back if needed
    if copy_back:
        if torch.is_tensor(orig_C):
            if orig_C.is_cuda:
                orig_C.view(-1)[:size].copy_(C_gpu[:size])
            else:
                orig_C.view(-1)[:size].copy_(C_gpu[:size].cpu())
        else:
            if isinstance(orig_C, np.ndarray):
                np.copyto(orig_C.ravel()[:size], C_gpu[:size].cpu().numpy())
            else:
                try:
                    tmp = C_gpu[:size].cpu().numpy()
                    orig_C[...] = tmp.reshape(orig_C.shape)
                except Exception as e:
                    raise RuntimeError("Unable to copy back results into original 'C' object") from e


# Optional simple self-test when executed as a script
if __name__ == "__main__":
    x = 1000
    A = torch.randn(x, dtype=torch.float32, device='cuda')
    B = torch.randn(x, dtype=torch.float32, device='cuda')
    C = torch.empty(x, dtype=torch.float32, device='cuda')
    triton_kernel(A, B, C, x)
    expected = A + B
    if not torch.allclose(C, expected):
        raise AssertionError("Mismatch between Triton kernel output and expected result")
    else:
        print("Triton kernel passed simple correctness test.")