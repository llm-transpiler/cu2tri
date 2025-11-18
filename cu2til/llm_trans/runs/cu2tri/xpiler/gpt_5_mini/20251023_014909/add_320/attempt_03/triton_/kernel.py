import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, TILE: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton kernel that performs elementwise A + B -> T_add.
    - TILE: number of valid lanes per program (matches CUDA block size = 320)
    - BLOCK: number of lanes for tl.arange; must be a power of two (e.g., 512)
    We use BLOCK (power-of-two) for tl.arange and mask out lanes >= TILE so behavior
    matches TILE lanes-per-program.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)                       # BLOCK must be power-of-two
    offs = pid * TILE + lane
    mask = (lane < TILE) & (offs < size)
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(T_add_ptr + offs, c, mask=mask)


def _next_power_of_two(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << ((x - 1).bit_length())


# Wrapper entry point (must be named exactly as requested and keep same signature)
def triton_kernel(A, B, C, size):
    """
    Entry point mirroring the original CUDA wrapper signature:
      triton_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : torch.cuda.FloatTensor (1-D) - device tensors
      size    : int - number of elements to process
    """
    # Configuration matching original CUDA wrapper
    TILE = 320  # original CUDA block size (threads per block)
    BLOCK = _next_power_of_two(TILE)  # must be power of two for tl.arange (e.g., 512)

    # Basic checks
    if not (torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(C)):
        raise TypeError("A, B, C must be torch tensors on CUDA with dtype=torch.float32")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("A, B, C must be float32 tensors")
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise ValueError("A, B, C must be CUDA tensors")
    if size < 0:
        raise ValueError("size must be non-negative")
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("A, B, and C must have at least 'size' elements")

    # Early exit
    if size == 0:
        return

    # Ensure 1-D contiguous buffers for pointer semantics
    if A.dim() != 1 or not A.is_contiguous():
        A = A.contiguous().view(-1)
    if B.dim() != 1 or not B.is_contiguous():
        B = B.contiguous().view(-1)
    if C.dim() != 1 or not C.is_contiguous():
        C = C.contiguous().view(-1)

    # Grid configuration: same logical grid as CUDA wrapper (ceil(size / TILE))
    num_blocks = (int(size) + TILE - 1) // TILE

    # Launch Triton kernel. TILE and BLOCK are compile-time constants.
    _triton_kernel_impl[(num_blocks,)](A, B, C, int(size), TILE=TILE, BLOCK=BLOCK)

    # Synchronize to match CUDA's default behavior if immediate results are expected.
    torch.cuda.synchronize()


# Optional simple test when run as a script
if __name__ == "__main__":
    size = 1024
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    triton_kernel(a, b, c, size)

    expected = a + b
    torch.cuda.synchronize()
    if torch.allclose(c, expected):
        print("Test passed: Triton kernel produced correct results.")
    else:
        diff = (c - expected).abs().max().item()
        print(f"Test failed: max abs difference = {diff}")