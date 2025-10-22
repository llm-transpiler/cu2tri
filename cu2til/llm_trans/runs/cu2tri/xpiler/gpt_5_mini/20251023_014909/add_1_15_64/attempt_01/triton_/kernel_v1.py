import triton
import triton.language as tl
import torch


# Triton kernel: elementwise add A + B -> T_add
# Must be named exactly _triton_kernel_impl per requirements.
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, T_add_ptr, size, BLOCK: tl.constexpr):
    """
    A_ptr, B_ptr, T_add_ptr : pointers to 1D float32 arrays
    size : number of elements to process (scalar)
    BLOCK : compile-time block size (number of elements handled per program)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    res = a + b
    tl.store(T_add_ptr + offs, res, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Entry-point wrapper with the same parameter signature as the original CUDA wrapper:
      cuda_kernel(float *A, float *B, float *C, int size)

    Parameters:
    - A, B, C : torch tensors or array-like objects. They will be converted to
                float32 CUDA contiguous 1D tensors if necessary.
    - size    : number of elements to process (int)

    Behavior: launches a Triton kernel with BLOCK == 960 (same as CUDA block size)
    """
    # Keep the same block size as the original CUDA kernel (blockDim.x == 960)
    BLOCK = 960

    # Ensure `size` is a Python int
    size = int(size)

    # Quick exit if there's nothing to do
    if size <= 0:
        return

    # Helper to ensure inputs are CUDA float32 contiguous tensors
    def _to_cuda_float32_tensor(x, name):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
        return x

    A_t = _to_cuda_float32_tensor(A, "A")
    B_t = _to_cuda_float32_tensor(B, "B")
    C_t = _to_cuda_float32_tensor(C, "C")

    # Flatten to 1D views for simple indexing (kernel uses linear offsets)
    A_flat = A_t.view(-1)
    B_flat = B_t.view(-1)
    C_flat = C_t.view(-1)

    # Ensure there are enough elements
    if A_flat.numel() < size or B_flat.numel() < size or C_flat.numel() < size:
        raise ValueError("A, B and C must have at least 'size' elements")

    # Compute grid (number of Triton program instances) matching CUDA's numBlocks
    num_blocks = (size + BLOCK - 1) // BLOCK
    if num_blocks <= 0:
        return

    # Launch Triton kernel
    _triton_kernel_impl[num_blocks](A_flat, B_flat, C_flat, size, BLOCK=BLOCK)


if __name__ == "__main__":
    # Simple test / demonstration (runs only if this file is executed directly).
    # Requires a CUDA-capable device and PyTorch with CUDA support.
    size = 10_000
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.empty(size, device="cuda", dtype=torch.float32)

    # Call the Triton wrapper (signature matches the original CUDA wrapper)
    triton_kernel(A, B, C, size)

    # Validate correctness
    if not torch.allclose(C, A + B):
        raise RuntimeError("Mismatch between Triton result and expected A + B")
    else:
        print("Triton kernel produced correct results for size =", size)