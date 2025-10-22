import torch
import triton
import triton.language as tl

# The original CUDA kernel processes up to 320 elements per block.
MAX_ELEM = 320

# Triton requires the tile size (threads per program) to be a power of two.
# Choose the largest power of two that does not exceed MAX_ELEM.
TILE_SIZE = 256

@triton.jit
def _triton_kernel_impl(
    A_ptr,  # *float
    B_ptr,  # *float
    C_ptr,  # *float
    n,      # i32, number of elements to compute (<= MAX_ELEM)
    TILE_SIZE: tl.constexpr,
):
    """
    Triton kernel reproducing the original CUDA kernel's behavior:
    For i in [0, n):
        C[i] = A[i] + B[i]
    The kernel does not use blockIdx.x; instead we tile the work across
    multiple programs (grid) when n > TILE_SIZE.
    """
    pid = tl.program_id(0)  # grid index
    offs = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offs < n

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Wrapper that launches the Triton kernel with the same signature as the
    original CUDA entry point.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    B : torch.Tensor
        Input tensor of shape (size,) and dtype torch.float32 on CUDA.
    C : torch.Tensor
        Output tensor of shape (size,) and dtype torch.float32 on CUDA.
    size : int
        Number of elements (mirrors the CUDA `size` argument). The original
        CUDA kernel only writes to the first 320 elements, so we cap the
        effective work size at `MAX_ELEM`.
    """
    # Sanity checks
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, \
        "All tensors must be torch.float32"
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous(), \
        "All tensors must be contiguous"
    assert A.shape == (size,) and B.shape == (size,) and C.shape == (size,), \
        "Tensor shapes must match the provided size"

    # Effective number of elements to compute (original kernel only processes up to MAX_ELEM)
    n = min(size, MAX_ELEM)

    # Grid: enough programs to cover `n` elements with TILE_SIZE threads each
    grid = lambda meta: (( (n + meta['TILE_SIZE'] - 1) // meta['TILE_SIZE'] ),)

    # Launch kernel
    _triton_kernel_impl[grid](
        A, B, C,
        n,
        TILE_SIZE=TILE_SIZE,
    )
    torch.cuda.synchronize()

# Simple sanity test
if __name__ == "__main__":
    size = 1024
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)
    triton_kernel(A, B, C, size)
    # Verify that the first MAX_ELEM elements are correct
    torch.testing.assert_close(C[:MAX_ELEM], A[:MAX_ELEM] + B[:MAX_ELEM])
    print("Triton kernel executed successfully.")