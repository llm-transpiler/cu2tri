import torch
import triton
import triton.language as tl

@triton.jit
def mha_inter_head_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    H: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """
    Triton kernel for inter-head multi-head attention.

    For each token in the batch and sequence, this kernel computes:
    1. S = Q @ K.T  (where Q, K are [H, D_HEAD] matrices for that token)
    2. P = softmax(S / sqrt(D_HEAD))
    3. O = P @ V      (where V is the [H, D_HEAD] matrix for that token)

    Grid: (B, N_CTX)
    Each program instance handles one token (i, j).
    """
    # 1. Get program IDs for batch and sequence dimensions
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. Create pointers for the [H, D_HEAD] block for the current token
    # Offsets for the H and D_HEAD dimensions
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D_HEAD)

    # Pointers for Q, K, V for the current token (i, j)
    # Shape of pointers will be [H, D_HEAD]
    q_ptrs = Q_ptr + (pid_b * stride_q_b + pid_n * stride_q_n +
                      offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d)
    k_ptrs = K_ptr + (pid_b * stride_k_b + pid_n * stride_k_n +
                      offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d)
    v_ptrs = V_ptr + (pid_b * stride_v_b + pid_n * stride_v_n +
                      offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d)

    # 3. Load Q, K, V blocks into SRAM.
    # q, k, v are now [H, D_HEAD] blocks in registers/SRAM.
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # 4. Compute score matrix: S = Q @ K.T
    # q: [H, D_HEAD], k: [H, D_HEAD] -> k.T: [D_HEAD, H]
    # scores: [H, H]
    # NOTE: We must cast to float32 for the matmul accumulation.
    q = q.to(tl.float32)
    k = k.to(tl.float32)
    scores = tl.dot(q, tl.trans(k))

    # 5. Compute scaled softmax
    # This fuses scaling, exp, sum, and division.
    scaling_factor = tl.math.rsqrt(float(D_HEAD))
    p = tl.softmax(scores * scaling_factor, axis=1) # Softmax over key heads

    # 6. Compute final output: O = P @ V
    # p: [H, H], v: [H, D_HEAD]
    # The dot product requires the inputs to have compatible types.
    p = p.to(v.dtype) # Cast probabilities to match V's dtype
    output = tl.dot(p, v)

    # 7. Write the output block back to global memory
    out_ptrs = Out_ptr + (pid_b * stride_o_b + pid_n * stride_o_n +
                        offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d)
    tl.store(out_ptrs, output)


def mha_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher function for the inter-head MHA Triton kernel.
    """
    # Input validation
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    B, N_CTX, H, D_HEAD = q.shape
    
    # Allocate output tensor
    output = torch.empty_like(q)
    
    # Define the grid
    grid = (B, N_CTX)
    
    # Launch the kernel
    mha_inter_head_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        H=H,
        D_HEAD=D_HEAD,
    )
    return output

# --- Verification ---

def mha_pytorch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A simple PyTorch implementation to verify the logic."""
    B, N_CTX, H, D_HEAD = q.shape
    
    # The operation is independent for each item in batch and sequence,
    # so we can use batch matrix multiplication.
    # Reshape to treat (B, N_CTX) as the batch dimension for matmul.
    q_b = q.view(B * N_CTX, H, D_HEAD)
    k_b = k.view(B * N_CTX, H, D_HEAD)
    v_b = v.view(B * N_CTX, H, D_HEAD)

    # S = Q @ K.T
    scores = torch.bmm(q_b, k_b.transpose(1, 2))
    
    # P = softmax(S / sqrt(D_HEAD))
    p = torch.nn.functional.softmax(scores / (D_HEAD**0.5), dim=-1)
    
    # O = P @ V
    output_b = torch.bmm(p, v_b)
    
    # Reshape back to original dimensions
    return output_b.view(B, N_CTX, H, D_HEAD)


if __name__ == "__main__":
    # Define problem dimensions from the CUDA code
    B = 4
    N_CTX = 2048
    H = 6
    D_HEAD = 256

    # Create random input tensors
    # Use float32 for direct comparison with the CUDA kernel's logic
    q = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)
    k = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)
    v = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)

    # Run Triton implementation
    output_triton = mha_triton(q, k, v)

    # Run PyTorch reference implementation
    output_pytorch = mha_pytorch_reference(q, k, v)

    # Compare results
    print(f"Triton implementation running on: {output_triton.device}")
    
    # Check for correctness
    is_close = torch.allclose(output_triton, output_pytorch, atol=1e-4, rtol=1e-4)
    print(f"Results are close: {is_close}")

    # For a more detailed check
    max_diff = (output_triton - output_pytorch).abs().max().item()
    print(f"Maximum absolute difference: {max_diff:.6f}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 1654, in arange
    return _semantic.arange(start, end)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 583, in arange
    raise ValueError("arange's range must be a power of 2")
ValueError: arange's range must be a power of 2

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 144, in <module>
    output_triton = mha_triton(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 93, in mha_triton
    mha_inter_head_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 390, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 339, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 83, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 27:13:
    3. O = P @ V      (where V is the [H, D_HEAD] matrix for that token)

    Grid: (B, N_CTX)
    Each program instance handles one token (i, j).
    """
    # 1. Get program IDs for batch and sequence dimensions
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. Create pointers for the [H, D_HEAD] block for the current token
    # Offsets for the H and D_HEAD dimensions
    offs_h = tl.arange(0, H)
             ^
'''