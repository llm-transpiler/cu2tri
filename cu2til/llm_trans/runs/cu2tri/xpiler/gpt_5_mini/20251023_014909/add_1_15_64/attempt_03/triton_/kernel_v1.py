import torch
import triton
import triton.language as tl

# Block size (threads per CUDA block in the original kernel)
_BLOCK = 960  # 30 warps * 32 lanes = 960 threads

@triton.jit(num_warps=30)
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    Triton kernel implementing elementwise add:
      T_add[i] = A[i] + B[i]  for i in [0, size)
    This kernel maps one Triton program instance to one CUDA block. Each instance
    processes a contiguous chunk of BLOCK elements:
      offs = pid * BLOCK + arange(0, BLOCK)
    A mask is used to avoid out-of-bounds accesses for the last (partial) block.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper with the same parameter signature as the original CUDA
    wrapper: (A, B, C, size)

    Parameters
    - A, B, C: torch.Tensor (will be cast/moved to CUDA float32 if needed)
    - size: int, number of elements to process

    The function configures the grid and launches the Triton kernel so that the
    behavior matches a CUDA launch with blockSize=960 and numBlocks=ceil(size/960).
    """
    # Basic argument validation / normalization to make the wrapper directly executable
    if not isinstance(size, int):
        size = int(size)

    if size <= 0:
        return  # nothing to do; matches behavior of launching zero blocks on CUDA

    # Make sure inputs are torch tensors
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C)

    # Move to CUDA if needed
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    if not C.is_cuda:
        C = C.cuda()

    # Ensure dtype is float32 (matches float* in the CUDA code)
    if A.dtype != torch.float32:
        A = A.to(dtype=torch.float32)
    if B.dtype != torch.float32:
        B = B.to(dtype=torch.float32)
    if C.dtype != torch.float32:
        C = C.to(dtype=torch.float32)

    # Ensure contiguous memory for efficient Triton access
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Ensure input tensors have enough elements
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Input tensors must have at least 'size' elements")

    # Compute grid configuration identically to the CUDA wrapper
    num_blocks = (size + _BLOCK - 1) // _BLOCK
    if num_blocks <= 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel (BLOCK passed as a constexpr)
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=_BLOCK)