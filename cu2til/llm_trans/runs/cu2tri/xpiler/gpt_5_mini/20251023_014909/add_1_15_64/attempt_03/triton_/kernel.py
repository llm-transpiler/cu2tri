import torch
import triton
import triton.language as tl

# Original CUDA block size
_BLOCK = 960
# Triton's tl.arange requires a power-of-two range. Choose the next power-of-two >= _BLOCK.
_BLOCK_VECT = 1 << (_BLOCK - 1).bit_length()  # 960 -> 1024


@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr, BLOCK_VECT: tl.constexpr):
    """
    Triton kernel that implements elementwise addition:
      T_add[i] = A[i] + B[i]  for i in [0, size)

    Because tl.arange requires a power-of-two length, we use BLOCK_VECT (a power of two,
    >= BLOCK) for the arange, and mask out the extra positions where v >= BLOCK.
    """
    pid = tl.program_id(0)
    v = tl.arange(0, BLOCK_VECT)                # BLOCK_VECT is power-of-two (constexpr)
    start = pid * BLOCK
    offs = start + v
    mask = (v < BLOCK) & (offs < size)          # mask out both beyond-block and beyond-size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper with the same parameter signature as the original CUDA wrapper:
      cuda_kernel(float *A, float *B, float *C, int size)

    This configures the grid to numBlocks = ceil(size / 960) and launches the Triton kernel.
    """
    # Normalize size
    if not isinstance(size, int):
        size = int(size)

    if size <= 0:
        return

    # Convert inputs to torch tensors if needed
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C)

    # Move to CUDA if necessary
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    if not C.is_cuda:
        C = C.cuda()

    # Ensure float32
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    if B.dtype != torch.float32:
        B = B.to(torch.float32)
    if C.dtype != torch.float32:
        C = C.to(torch.float32)

    # Ensure contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Ensure enough elements
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Input tensors must have at least 'size' elements")

    # Compute grid exactly like the CUDA wrapper
    num_blocks = (size + _BLOCK - 1) // _BLOCK
    if num_blocks <= 0:
        return

    grid = (num_blocks,)

    # Launch Triton kernel, passing BLOCK and BLOCK_VECT as constexpr parameters
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=_BLOCK, BLOCK_VECT=_BLOCK_VECT)