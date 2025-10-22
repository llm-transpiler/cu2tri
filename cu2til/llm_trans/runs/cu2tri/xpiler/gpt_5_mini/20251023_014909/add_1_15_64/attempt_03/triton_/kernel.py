import torch
import triton
import triton.language as tl

# Block size matching the CUDA __launch_bounds__(960) and the original blockDim.x = 960
_BLOCK = 960


@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    Triton kernel that computes:
      T_add[i] = A[i] + B[i]  for i in [0, size)

    One Triton program instance maps to one CUDA block and processes BLOCK contiguous elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(T_add_ptr + offs, a + b, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper with the same parameter signature as the original CUDA wrapper:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
      A, B, C : tensors or array-like (will be converted to torch.cuda.FloatTensor)
      size    : int number of elements to process

    This configures the grid to numBlocks = ceil(size / 960) and launches the Triton kernel.
    """
    # Normalize size to int
    if not isinstance(size, int):
        size = int(size)

    # Nothing to do
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

    # Force float32 (matches float* in CUDA code)
    if A.dtype != torch.float32:
        A = A.to(torch.float32)
    if B.dtype != torch.float32:
        B = B.to(torch.float32)
    if C.dtype != torch.float32:
        C = C.to(torch.float32)

    # Ensure contiguous memory for efficient Triton access
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Ensure there are enough elements
    if A.numel() < size or B.numel() < size or C.numel() < size:
        raise ValueError("Input tensors must have at least 'size' elements")

    # Compute grid configuration like the original CUDA wrapper
    num_blocks = (size + _BLOCK - 1) // _BLOCK
    if num_blocks <= 0:
        return

    grid = (num_blocks,)

    # Launch the Triton kernel. BLOCK is passed as a constexpr parameter.
    _triton_kernel_impl[grid](A, B, C, size, BLOCK=_BLOCK)