import torch
import triton
import triton.language as tl

# Make the 2304 limit a Triton constexpr so it can be used inside @triton.jit
_LIMIT = tl.constexpr(2304)

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, BLOCK: tl.constexpr):
    """
    Triton kernel mirroring the CUDA kernel:
      if (global_idx < 2304) T_add[global_idx] = A[global_idx] + B[global_idx];
    Each program handles BLOCK consecutive elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < _LIMIT
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Host wrapper matching the original CUDA host signature:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.Tensor on CUDA (any shape). They will be flattened for pointer-style access.
      size     : int, used to compute number of blocks exactly as the CUDA host did.
    """
    # Basic checks
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor instances")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be float32 tensors")

    # Prepare flattened / contiguous tensors for the kernel.
    # For A and B we can make contiguous copies if needed (no need to copy back).
    if A.dim() == 1 and A.is_contiguous():
        A_flat = A.view(-1)
    else:
        A_flat = A.contiguous().view(-1)

    if B.dim() == 1 and B.is_contiguous():
        B_flat = B.view(-1)
    else:
        B_flat = B.contiguous().view(-1)

    # For C, if it's not a contiguous 1-D view we make a contiguous buffer and copy back after the kernel.
    need_copy_back = False
    if C.dim() == 1 and C.is_contiguous():
        C_flat = C.view(-1)
    else:
        C_flat = C.contiguous().view(-1)
        need_copy_back = True

    # Compute grid exactly like the original CUDA host code
    BLOCK = 1024
    num_blocks = (int(size) + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return
    grid = (num_blocks,)

    # Launch Triton kernel. BLOCK is a constexpr argument.
    _triton_kernel_impl[grid](A_flat, B_flat, C_flat, BLOCK=BLOCK)

    # If we created a temporary contiguous C buffer, copy results back into the original C tensor.
    if need_copy_back:
        # reshape to original shape and copy
        C.copy_(C_flat.view(C.shape))


# Optional basic self-test when run as a script
if __name__ == "__main__":
    device = "cuda"
    H, W = 18, 128  # 18*128 = 2304, matches the original kernel's hard-coded limit
    size = H * W
    A = torch.arange(size, dtype=torch.float32, device=device).reshape(H, W)
    B = (torch.arange(size, dtype=torch.float32, device=device) * 2.0).reshape(H, W)
    C = torch.empty((H, W), dtype=torch.float32, device=device)

    triton_kernel(A, B, C, size)

    expected = A + B
    if not torch.allclose(C, expected):
        diff = (C - expected).abs().max().item()
        raise RuntimeError(f"Mismatch after Triton kernel launch. max diff = {diff}")
    else:
        print("Triton kernel produced correct results for the test.")