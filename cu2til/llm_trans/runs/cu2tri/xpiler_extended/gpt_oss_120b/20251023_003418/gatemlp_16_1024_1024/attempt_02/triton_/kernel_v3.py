import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel implementing the original CUDA gatemlp_kernel
# ----------------------------------------------------------------------
@triton.jit
def _triton_kernel_impl(
    x_ptr, a_ptr, b_ptr, out_ptr,          # pointers
    batch, K, N,                           # scalar arguments
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute for each (row, col):
        acc1 = sum_k x[row, k] * a[k, col]
        acc2 = sum_k x[row, k] * b[k, col]
        silu = acc1 / (1 + exp(-acc1))
        out  = silu * acc2
    Accumulations are performed in fp32 (sufficient for the original fp64
    computation while avoiding unsupported fp64 MMA on Hopper/H800).
    """
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # N dimension

    # ------------------------------------------------------------------
    # Row / column indices for this program instance
    # ------------------------------------------------------------------
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    batch_mask = row < batch          # [BLOCK_M]
    N_mask = col < N                  # [BLOCK_N]
    store_mask = batch_mask[:, None] & N_mask[None, :]   # [BLOCK_M, BLOCK_N]

    # ------------------------------------------------------------------
    # Accumulators (fp32)
    # ------------------------------------------------------------------
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Tiled loop over K dimension
    # ------------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)          # [BLOCK_K]
        K_mask = cur_k < K                         # [BLOCK_K]

        # Load x: shape (BLOCK_M, BLOCK_K)
        x_index = row[:, None] * K + cur_k[None, :]   # broadcast to [BLOCK_M, BLOCK_K]
        x = tl.load(
            x_ptr + x_index,
            mask=batch_mask[:, None] & K_mask[None, :],
            other=0.0,
        )
        x = x.to(tl.float32)

        # Load a: shape (BLOCK_K, BLOCK_N)
        a_index = cur_k[:, None] * N + col[None, :]   # [BLOCK_K, BLOCK_N]
        a = tl.load(
            a_ptr + a_index,
            mask=K_mask[:, None] & N_mask[None, :],
            other=0.0,
        )
        a = a.to(tl.float32)

        # Load b: shape (BLOCK_K, BLOCK_N)
        b = tl.load(
            b_ptr + a_index,   # same indexing as a
            mask=K_mask[:, None] & N_mask[None, :],
            other=0.0,
        )
        b = b.to(tl.float32)

        # Accumulate partial matrix products
        acc1 += tl.dot(x, a)   # (BLOCK_M, BLOCK_N)
        acc2 += tl.dot(x, b)

    # ------------------------------------------------------------------
    # Apply SiLU activation and write back
    # ------------------------------------------------------------------
    silu = acc1 / (1.0 + tl.exp(-acc1))
    out = silu * acc2                     # still fp32
    out_index = row[:, None] * N + col[None, :]
    tl.store(out_ptr + out_index, out, mask=store_mask)


# ----------------------------------------------------------------------
# Python wrapper that mirrors the original CUDA entry point
# ----------------------------------------------------------------------
def triton_kernel(x, a, b, output, batch, K, N):
    """
    Launches the Triton implementation of the GATEMLP kernel.

    Parameters
    ----------
    x : torch.Tensor (batch, K), dtype=torch.float16, CUDA
    a : torch.Tensor (K, N), dtype=torch.float16, CUDA
    b : torch.Tensor (K, N), dtype=torch.float16, CUDA
    output : torch.Tensor (batch, N), dtype=torch.float32, CUDA (pre‑allocated)
    batch, K, N : int
        Dimensions matching the tensor shapes.
    """
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert x.is_cuda and a.is_cuda and b.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.float16 and a.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be half (float16)"
    assert output.dtype == torch.float32, "Output must be float32"
    assert x.shape == (batch, K)
    assert a.shape == (K, N)
    assert b.shape == (K, N)
    assert output.shape == (batch, N)

    # Ensure contiguous layout
    x = x.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    output = output.contiguous()

    # ------------------------------------------------------------------
    # Tunable block sizes (chosen for Hopper/H800)
    # ------------------------------------------------------------------
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Compute grid dimensions (ceil division)
    grid_m = triton.cdiv(batch, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    # Launch the kernel
    _triton_kernel_impl[(grid_m, grid_n)](
        x,
        a,
        b,
        output,
        batch,
        K,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Synchronize to guarantee completion before returning
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Simple correctness test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example dimensions (feel free to adjust)
    batch = 128
    K = 256
    N = 512

    # Allocate inputs
    x = torch.randn(batch, K, device="cuda", dtype=torch.float16)
    a = torch.randn(K, N, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    out = torch.empty(batch, N, device="cuda", dtype=torch.float32)

    # Reference implementation (double precision)
    x_f64 = x.double()
    a_f64 = a.double()
    b_f64 = b.double()
    acc1_ref = torch.matmul(x_f64, a_f64)          # (batch, N) fp64
    acc2_ref = torch.matmul(x_f64, b_f64)          # (batch, N) fp64
    silu_ref = acc1_ref / (1.0 + torch.exp(-acc1_ref))
    out_ref = (silu_ref * acc2_ref).float()       # back to fp32

    # Run Triton kernel
    triton_kernel(x, a, b, out, batch, K, N)

    # Verify correctness
    max_abs_err = (out - out_ref).abs().max().item()
    print(f"Maximum absolute error vs reference: {max_abs_err:.3e}")