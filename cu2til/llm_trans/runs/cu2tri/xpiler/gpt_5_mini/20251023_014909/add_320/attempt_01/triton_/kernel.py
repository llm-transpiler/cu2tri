import triton
import triton.language as tl
import torch
import numpy as np

# Triton kernel implementation (must be named exactly as requested)
@triton.jit
def _triton_kernel_impl(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    """
    Each program processes BLOCK elements. This kernel performs:
      for i in range(BLOCK):
        idx = program_id(0) * BLOCK + i
        if idx < size:
            C[idx] = A[idx] + B[idx]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)


def triton_kernel(A, B, C, size):
    """
    Wrapper that matches the CUDA kernel signature: triton_kernel(A, B, C, size)
    - A, B, C can be torch tensors (preferred), numpy arrays or array-like.
    - If inputs are not on CUDA or not float32/contiguous, they will be moved/copied.
    - The result is written back to the original C when C is a CPU tensor/numpy array.
    """
    BLOCK = 320  # match the CUDA block size

    # Validate and normalize size
    size = int(size)
    if size <= 0:
        return

    # Helper to convert arbitrary array-like to a torch tensor (CPU) if needed
    def _to_torch_tensor(x):
        if torch.is_tensor(x):
            return x
        # numpy arrays -> torch tensor sharing memory
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        # other array-like -> as_tensor
        return torch.as_tensor(x)

    # Prepare A
    A_in = _to_torch_tensor(A)
    if not (A_in.is_cuda and A_in.dtype == torch.float32 and A_in.is_contiguous()):
        A_gpu = A_in.contiguous().to(device='cuda', dtype=torch.float32)
    else:
        A_gpu = A_in.contiguous()  # already suitable; keep as-is
    A_gpu = A_gpu.view(-1)

    # Prepare B
    B_in = _to_torch_tensor(B)
    if not (B_in.is_cuda and B_in.dtype == torch.float32 and B_in.is_contiguous()):
        B_gpu = B_in.contiguous().to(device='cuda', dtype=torch.float32)
    else:
        B_gpu = B_in.contiguous()
    B_gpu = B_gpu.view(-1)

    # Prepare C and remember whether we need to copy back
    C_in = _to_torch_tensor(C)
    copy_back = False
    orig_C = C  # keep original to copy back if necessary

    if torch.is_tensor(C_in) and C_in.is_cuda and C_in.dtype == torch.float32 and C_in.is_contiguous():
        C_gpu = C_in.view(-1)
        copy_back = False
    else:
        # create a GPU buffer for computation
        C_gpu_tmp = C_in.contiguous().to(device='cuda', dtype=torch.float32)
        C_gpu = C_gpu_tmp.view(-1)
        copy_back = True

    # Sanity check: ensure inputs have enough elements
    if A_gpu.numel() < size or B_gpu.numel() < size or C_gpu.numel() < size:
        raise ValueError(
            f"Input tensors must have at least 'size' elements: "
            f"got A={A_gpu.numel()}, B={B_gpu.numel()}, C={C_gpu.numel()}, size={size}"
        )

    # Launch Triton kernel
    num_blocks = (size + BLOCK - 1) // BLOCK
    if num_blocks > 0:
        _triton_kernel_impl[(num_blocks,)](A_gpu, B_gpu, C_gpu, size, BLOCK=BLOCK)

    # Copy results back to original C if necessary
    if copy_back:
        # If original was a torch tensor (CPU), copy into it
        if torch.is_tensor(orig_C):
            if orig_C.is_cuda:
                # unlikely path: orig_C is CUDA but we created a new GPU buffer (different dtype/contiguity)
                orig_C.view(-1)[:size].copy_(C_gpu[:size])
            else:
                # copy GPU -> CPU
                orig_C.view(-1)[:size].copy_(C_gpu[:size].cpu())
        else:
            # assume numpy array or array-like supporting item assignment
            orig = orig_C
            if isinstance(orig, np.ndarray):
                # Overwrite in-place
                np.copyto(orig.ravel()[:size], C_gpu[:size].cpu().numpy())
            else:
                # Fallback: try to set via buffer from torch
                try:
                    tmp = C_gpu[:size].cpu().numpy()
                    orig[...] = tmp.reshape(orig.shape)  # may raise if shapes mismatch
                except Exception as e:
                    raise RuntimeError("Unable to copy back results into original 'C' object") from e


# Optional simple self-test when executed as a script
if __name__ == "__main__":
    # Small correctness test
    x = 1000
    A = torch.randn(x, dtype=torch.float32, device='cuda')
    B = torch.randn(x, dtype=torch.float32, device='cuda')
    C = torch.empty(x, dtype=torch.float32, device='cuda')
    triton_kernel(A, B, C, x)
    # verify
    expected = (A + B)
    if not torch.allclose(C, expected):
        raise AssertionError("Mismatch between Triton kernel output and expected result")
    else:
        print("Triton kernel passed simple correctness test.")