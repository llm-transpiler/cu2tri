import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Helper function to find the next power of 2
def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

@triton.jit
def mha_inter_head_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    H: tl.constexpr,          # Original number of heads (e.g., 6)
    H_PAD: tl.constexpr,      # Padded number of heads (e.g., 16)
    D_HEAD: tl.constexpr,
):
    """
    Triton kernel for inter-head multi-head attention.
    Handles non-power-of-two H by padding to H_PAD, ensuring H_PAD >= 16.
    """
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_h = tl.arange(0, H_PAD)
    offs_d = tl.arange(0, D_HEAD)

    h_mask = offs_h < H

    q_ptrs = Q_ptr + (pid_b * stride_q_b + pid_n * stride_q_n +
                      offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d)
    k_ptrs = K_ptr + (pid_b * stride_k_b + pid_n * stride_k_n +
                      offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d)
    v_ptrs = V_ptr + (pid_b * stride_v_b + pid_n * stride_v_n +
                      offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d)

    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=h_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=h_mask[:, None], other=0.0)

    q = q.to(tl.float32)
    k = k.to(tl.float32)
    scores = tl.dot(q, tl.trans(k))

    scaling_factor = tl.math.rsqrt(float(D_HEAD))
    scores *= scaling_factor
    
    scores = tl.where(h_mask[None, :], scores, -float('inf'))

    p = tl.softmax(scores, axis=1)

    p = p.to(v.dtype)
    output = tl.dot(p, v)

    out_ptrs = Out_ptr + (pid_b * stride_o_b + pid_n * stride_o_n +
                        offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d)
    tl.store(out_ptrs, output, mask=h_mask[:, None])


def mha_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher function for the inter-head MHA Triton kernel.
    Handles padding for non-power-of-two head dimension.
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    B, N_CTX, H, D_HEAD = q.shape
    
    # --- THIS IS THE FIX ---
    # Determine padding for the H dimension, ensuring it's at least 16.
    H_PAD = max(16, next_power_of_2(H))
    
    if H_PAD != H:
        q_pad = F.pad(q, (0, 0, 0, H_PAD - H))
        k_pad = F.pad(k, (0, 0, 0, H_PAD - H))
        v_pad = F.pad(v, (0, 0, 0, H_PAD - H))
    else:
        q_pad, k_pad, v_pad = q, k, v

    output_pad = torch.empty_like(q_pad)
    
    grid = (B, N_CTX)
    
    mha_inter_head_kernel[grid](
        q_pad, k_pad, v_pad, output_pad,
        q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
        k_pad.stride(0), k_pad.stride(1), k_pad.stride(2), k_pad.stride(3),
        v_pad.stride(0), v_pad.stride(1), v_pad.stride(2), v_pad.stride(3),
        output_pad.stride(0), output_pad.stride(1), output_pad.stride(2), output_pad.stride(3),
        H=H,
        H_PAD=H_PAD,
        D_HEAD=D_HEAD,
    )
    
    if H_PAD != H:
        return output_pad[:, :, :H, :]
    else:
        return output_pad

# --- Verification (unchanged) ---

def mha_pytorch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A simple PyTorch implementation to verify the logic."""
    B, N_CTX, H, D_HEAD = q.shape
    q_b = q.view(B * N_CTX, H, D_HEAD)
    k_b = k.view(B * N_CTX, H, D_HEAD)
    v_b = v.view(B * N_CTX, H, D_HEAD)
    scores = torch.bmm(q_b, k_b.transpose(1, 2))
    p = torch.nn.functional.softmax(scores / (D_HEAD**0.5), dim=-1)
    output_b = torch.bmm(p, v_b)
    return output_b.view(B, N_CTX, H, D_HEAD)


if __name__ == "__main__":
    B, N_CTX, H, D_HEAD = 4, 2048, 6, 256
    q = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)
    k = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)
    v = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=torch.float32)

    output_triton = mha_triton(q, k, v)
    output_pytorch = mha_pytorch_reference(q, k, v)

    print(f"Triton implementation running on: {output_triton.device}")
    is_close = torch.allclose(output_triton, output_pytorch, atol=1e-4, rtol=1e-4)
    print(f"Results are close: {is_close}")
    max_diff = (output_triton - output_pytorch).abs().max().item()
    print(f"Maximum absolute difference: {max_diff:.6f}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 124, in <module>
    output_triton = mha_triton(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 88, in mha_triton
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
triton.compiler.errors.CompilationError: at 43:8:
    v = tl.load(v_ptrs, mask=h_mask[:, None], other=0.0)

    q = q.to(tl.float32)
    k = k.to(tl.float32)
    scores = tl.dot(q, tl.trans(k))

    scaling_factor = tl.math.rsqrt(float(D_HEAD))
    scores *= scaling_factor

    scores = tl.where(h_mask[None, :], scores, -float('inf'))

    p = tl.softmax(scores, axis=1)
        ^
TypeError("softmax() got an unexpected keyword argument 'axis'")
'''