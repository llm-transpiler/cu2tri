import torch
import triton
import triton.language as tl

@triton.jit
def _triton_kernel_impl(
    A_ptr, B_ptr, C_ptr, size,
    BLOCK_SIZE: tl.constexpr,
    OUTER_ITER: tl.constexpr
):
    pid = tl.program_id(0)               # block index (equivalent to blockIdx.x)
    total_blocks = tl.num_programs(0)    # total number of blocks in the grid
    offs = tl.arange(0, BLOCK_SIZE)     # thread indices within the block (threadIdx.x)
    for outer in range(OUTER_ITER):
        # Global linear index for each thread, matching the CUDA formulation:
        # index = outer * (grid_x * BLOCK_SIZE) + blockIdx.x * BLOCK_SIZE + threadIdx.x
        idx = outer * (total_blocks * BLOCK_SIZE) + pid * BLOCK_SIZE + offs
        mask = idx < size                # guard against out‑of‑bounds accesses
        a = tl.load(A_ptr + idx, mask=mask, other=0.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        tl.store(C_ptr + idx, a + b, mask=mask)

def triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, size: int):
    """
    Triton equivalent of the original CUDA kernel launch.
    Parameters:
        A (torch.Tensor): Input tensor A (float32, CUDA).
        B (torch.Tensor): Input tensor B (float32, CUDA).
        C (torch.Tensor): Output tensor C (float32, CUDA).
        size (int): Number of elements to process.
    """
    # Validate inputs
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise RuntimeError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise RuntimeError("All tensors must be torch.float32")
    # Ensure contiguous layout for pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    # Kernel launch configuration (mirrors the CUDA launch parameters)
    BLOCK_SIZE = 1024
    OUTER_ITER = 8
    total_per_block = BLOCK_SIZE * OUTER_ITER
    # Compute grid size to cover `size` elements (ceil division)
    grid = ((size + total_per_block - 1) // total_per_block,)
    # Launch the Triton kernel
    _triton_kernel_impl[grid](
        A, B, C, size,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTER_ITER=OUTER_ITER,
        num_warps=BLOCK_SIZE // 32,   # 32 warps for 1024 threads per block
    )
    # Optional synchronization for debugging (comment out in production)
    # torch.cuda.synchronize()