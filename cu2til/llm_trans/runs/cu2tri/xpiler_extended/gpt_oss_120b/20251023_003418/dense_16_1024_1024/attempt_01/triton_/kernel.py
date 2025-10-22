import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    a_ptr, b_ptr, c_ptr, bias_ptr,               # pointers
    M, K, N,                                      # matrix dimensions
    stride_am, stride_ak,                         # strides of A
    stride_bk, stride_bn,                         # strides of B
    stride_cm, stride_cn,                         # strides of C (output)
    bias_stride,                                  # stride of bias
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    """
    Compute C = A @ B + bias
    A : (M, K)  half
    B : (K, N)  half
    bias : (N,) float32
    C : (M, N) float32
    """
    pid_m = tl.program_id(axis=1)   # block row
    pid_n = tl.program_id(axis=0)   # block column

    # Block start indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Out‑of‑bounds masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------
    # K‑dimension tiling
    # ------------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers to the current tiles
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        # Load tiles (half precision). `other=0` provides a zero for masked elements.
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0)

        # Cast to FP32 before the dot product (matches __half2float behavior)
        a_f32 = a.to(tl.float32)
        b_f32 = b.to(tl.float32)

        # Matrix multiply‑accumulate
        acc += tl.dot(a_f32, b_f32)

    # ------------------------------------------------------------------
    # Add bias (broadcast across rows)
    # ------------------------------------------------------------------
    bias = tl.load(bias_ptr + offs_n * bias_stride,
                    mask=mask_n,
                    other=0.0)                     # bias is already FP32
    acc += bias[None, :]                     # broadcast to (BLOCK_M, BLOCK_N)

    # ------------------------------------------------------------------
    # Write result
    # ------------------------------------------------------------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs,
             acc,
             mask=mask_m[:, None] & mask_n[None, :])

# ----------------------------------------------------------------------
# Python wrapper (entry point)
# ----------------------------------------------------------------------
def triton_kernel(a, b, bias, output, M, K, N):
    """
    Launches the Triton kernel with the same signature as the original CUDA kernel.

    Parameters
    ----------
    a : torch.Tensor
        (M, K) half‑precision matrix.
    b : torch.Tensor
        (K, N) half‑precision matrix.
    bias : torch.Tensor
        (N,) float‑32 vector.
    output : torch.Tensor
        (M, N) float‑32 matrix to be filled.
    M, K, N : int
        Matrix dimensions.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (a.is_cuda and b.is_cuda and bias.is_cuda and output.is_cuda):
        raise RuntimeError("All tensors must be on CUDA")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise RuntimeError("A and B must be torch.float16")
    if bias.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("Bias and output must be torch.float32")

    # Ensure contiguous layout (required for correct stride handling)
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    output = output.contiguous()

    # --------------------------------------------------------------
    # Extract strides (in elements, not bytes)
    # --------------------------------------------------------------
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = output.stride()
    bias_stride = bias.stride(0)

    # --------------------------------------------------------------
    # Tunable block sizes – chosen for Hopper‑class GPUs (e.g., H800)
    # --------------------------------------------------------------
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 3

    # --------------------------------------------------------------
    # Grid configuration
    # --------------------------------------------------------------
    grid = ((N + BLOCK_N - 1) // BLOCK_N,
            (M + BLOCK_M - 1) // BLOCK_M)

    # --------------------------------------------------------------
    # Kernel launch
    # --------------------------------------------------------------
    _triton_kernel_impl[grid](
        a,
        b,
        output,
        bias,
        M,
        K,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        bias_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

# ----------------------------------------------------------------------
# Simple correctness test (run only when executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float32, device="cuda")
    out = torch.empty(M, N, dtype=torch.float32, device="cuda")

    # Run Triton implementation
    triton_kernel(a, b, bias, out, M, K, N)

    # Reference result (float32 matmul + bias)
    ref = a.float() @ b.float() + bias
    max_err = (out - ref).abs().max().item()
    print(f"Max absolute error vs PyTorch reference: {max_err:.3e}")