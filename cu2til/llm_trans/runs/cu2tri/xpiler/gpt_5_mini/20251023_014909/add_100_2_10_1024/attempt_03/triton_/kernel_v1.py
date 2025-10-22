import torch
import triton
import triton.language as tl

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, N, BLOCK: tl.constexpr):
    """
    Triton kernel that mirrors the original CUDA kernel's behavior:
    - Each Triton program corresponds to one CUDA block (blockIdx.x).
    - Within each program we emulate threadIdx.x with a vector of length BLOCK.
    - We loop `outer` from 0..7 with a stride equal to grid_size*BLOCK to cover the
      same global indices as the CUDA kernel.
    - Stores are masked by idx < N.
    """
    pid = tl.program_id(0)                       # CUDA blockIdx.x
    offs = tl.arange(0, BLOCK)                   # emulates threadIdx.x from 0..BLOCK-1
    base = pid * BLOCK + offs                    # base linear index for this "thread"
    stride = BLOCK * tl.num_programs(0)          # gridDim.x * blockDim.x
    for outer in range(8):                       # original kernel loops outer in [0,8)
        idx = base + outer * stride              # compute global index
        mask = idx < N                           # mask for valid indices (idx < 2048000)
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        c = a + b
        tl.store(C_ptr + idx, c, mask=mask)

# Wrapper entry point (must be named triton_kernel and have same signature as CUDA wrapper)
def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Entry point that mirrors the CUDA wrapper:
      extern "C" void cuda_kernel(float *A, float *B, float *C, int size)
    Behavior matches the original CUDA wrapper: it launches the kernel with
    blockDim.x=1024 and gridDim.x=256. The runtime 'size' parameter is accepted
    for signature compatibility but the kernel uses the same hard-coded bound
    check as the CUDA kernel (2048000), matching original behavior.
    """
    # Basic checks to help users avoid common mistakes
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, C must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("A, B, C must be torch.float32 tensors")
    if A.numel() != B.numel() or A.numel() != C.numel():
        raise ValueError("A, B, C must have the same number of elements")
    # Match CUDA launch configuration from the original code
    BLOCK = 1024      # blockDim.x
    NUM_BLOCKS = 256  # gridDim.x

    # Original CUDA kernel used a hard-coded bound of 2048000 for the index check.
    # We replicate that exactly to preserve original behavior.
    N = 2048000

    # Ensure contiguous tensors (Triton works best with contiguous inputs)
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Launch Triton kernel: one program per CUDA block (NUM_BLOCKS)
    # BLOCK is provided as a constexpr to the Triton kernel.
    _triton_kernel_impl[(NUM_BLOCKS,)](A, B, C, N, BLOCK=BLOCK)