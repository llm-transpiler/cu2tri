import torch
import triton
import triton.language as tl

# Compile-time constants matching the original CUDA kernel
BLOCK_SIZE = 1024  # threads per block (launch bounds)
BOUND = 2304       # hard‑coded upper bound used in the CUDA kernel

@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr,
                        BLOCK_SIZE: tl.constexpr,
                        BOUND: tl.constexpr):
    """
    Triton implementation of the original CUDA kernel.
    Performs C[i] = A[i] + B[i] for i < BOUND.
    """
    pid = tl.program_id(0)                     # block index (gridDim.x)
    offsets = tl.arange(0, BLOCK_SIZE)         # thread indices within the block
    idx = pid * BLOCK_SIZE + offsets            # global linear index

    mask = idx < BOUND                          # bound check (2304)

    a = tl.load(A_ptr + idx, mask=mask, other=0.0)
    b = tl.load(B_ptr + idx, mask=mask, other=0.0)
    tl.store(C_ptr + idx, a + b, mask=mask)


def triton_kernel(A: torch.Tensor,
                  B: torch.Tensor,
                  C: torch.Tensor,
                  size: int):
    """
    Wrapper that mimics the original CUDA entry point:
        void cuda_kernel(float *A, float *B, float *C, int size)

    Parameters
    ----------
    A : torch.Tensor
        Input tensor A (float32, CUDA)
    B : torch.Tensor
        Input tensor B (float32, CUDA)
    C : torch.Tensor
        Output tensor C (float32, CUDA)
    size : int
        Number of elements (used only to compute grid dimensions)
    """
    # Sanity checks (mirroring typical CUDA expectations)
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32")

    # Compute grid size exactly as in the CUDA host code
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    _triton_kernel_impl[grid](A, B, C,
                              BLOCK_SIZE=BLOCK_SIZE,
                              BOUND=BOUND)


# Example usage / sanity check
if __name__ == "__main__":
    # Use the same bound as the original kernel for a fair test
    size = BOUND
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    triton_kernel(A, B, C, size)

    # Verify correctness against PyTorch's native addition
    torch.testing.assert_allclose(C, A + B, atol=1e-6, rtol=1e-6)
    print("Triton kernel executed successfully and results match.")