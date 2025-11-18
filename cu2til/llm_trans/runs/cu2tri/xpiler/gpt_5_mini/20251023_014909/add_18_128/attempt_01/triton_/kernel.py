import torch
import triton
import triton.language as tl

# Triton kernel implementation: element-wise add A + B -> T_add
# Must be named exactly as requested.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, check_n, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_add_ptr : pointers to 1D float32 tensors (flattened)
    check_n : integer used for the bounds check (the original CUDA kernel used the hard-coded 2304)
    BLOCK : compile-time block size (should be 1024 to match the CUDA kernel)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)       # no dtype kw (not accepted by this Triton version)
    mask = offs < check_n
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper entry point that matches the CUDA kernel signature:
      triton_kernel(A, B, C, size)
    where A, B, C are torch.cuda.FloatTensor (possibly multi-dimensional) and size is an int.

    Behavior mirrors the original CUDA:
      - Uses block size 1024
      - Computes numBlocks = (size + 1024 - 1) // 1024
      - The kernel performs a bounds check against the constant 2304 (replicating the original kernel)
    """
    # Basic validations
    if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and isinstance(C, torch.Tensor)):
        raise TypeError("A, B, C must be torch.Tensor")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if not (A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32):
        raise TypeError("A, B, C must be torch.float32 tensors")
    if not isinstance(size, int):
        raise TypeError("size must be an int")

    # Flatten inputs to 1-D views when possible (avoid copies if tensors are already compatible)
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    # If C_flat does not share the same storage pointer as the original C (i.e., reshape created a copy),
    # or if it's non-contiguous, we'll write into a contiguous temporary and copy back afterwards to preserve semantics.
    need_copy_back = (C_flat.data_ptr() != C.data_ptr()) or (not C_flat.is_contiguous())

    # Ensure there are enough elements for 'size' as used to compute the grid
    if A_flat.numel() < size or B_flat.numel() < size or C_flat.numel() < size:
        raise ValueError("A, B, C must have at least 'size' elements when flattened")

    # Prepare buffers for Triton kernel (must be contiguous flattened tensors)
    if not A_flat.is_contiguous():
        A_flat = A_flat.contiguous()
    if not B_flat.is_contiguous():
        B_flat = B_flat.contiguous()

    if need_copy_back:
        # create a contiguous buffer to write results into
        write_buf = C_flat.contiguous()
    else:
        write_buf = C_flat  # already contiguous and shares storage

    # Match CUDA launch configuration
    BLOCK = 1024
    num_blocks = (size + BLOCK - 1) // BLOCK
    grid = (num_blocks,)

    # Launch Triton kernel with the hard-coded check value 2304 to match original CUDA implementation
    _triton_kernel_impl[grid](A_flat, B_flat, write_buf, 2304, BLOCK=BLOCK)

    # If we wrote into a temporary buffer, copy results back into original C
    if need_copy_back:
        # reshape to original shape and copy
        C.copy_(write_buf.view(C.shape))


# Optional self-test when run as a script
if __name__ == "__main__":
    # The original CUDA kernel uses a hard-coded 2304 bound; use a matching test size.
    size = 2304
    # Create sample 2-D tensors (e.g., 18 x 128) to ensure flattening works
    A = torch.randn(18, 128, device="cuda", dtype=torch.float32)
    B = torch.randn(18, 128, device="cuda", dtype=torch.float32)
    C = torch.empty(18, 128, device="cuda", dtype=torch.float32)

    triton_kernel(A, B, C, size)
    torch.cuda.synchronize()

    expected = A + B
    if not torch.allclose(C, expected):
        diff_max = (C - expected).abs().max().item()
        raise RuntimeError(f"Result mismatch (max abs diff = {diff_max})")
    else:
        print("Triton kernel succeeded: C == A + B")