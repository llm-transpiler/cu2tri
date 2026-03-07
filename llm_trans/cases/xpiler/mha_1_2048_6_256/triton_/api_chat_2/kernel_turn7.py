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
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    D_HEAD: tl.constexpr,
):
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

    # --- PRECISION FIX: Force all calculations to float32 ---
    q = q.to(tl.float32)
    k = k.to(tl.float32)
    v = v.to(tl.float32)

    scores = tl.dot(q, tl.trans(k))
    scores *= tl.math.rsqrt(float(D_HEAD))
    
    key_mask = h_mask[None, :]
    scores = tl.where(key_mask, scores, -float('inf'))
    
    query_mask = h_mask[:, None]
    # This where is not strictly needed if the next one works, but it's safer
    scores = tl.where(query_mask, scores, 0.0)
    
    p = tl.softmax(scores)

    p = tl.where(query_mask, p, 0.0)

    # p and v are both float32 now
    output = tl.dot(p, v)

    # Store the result, casting back to the original output type
    out_ptrs = Out_ptr + (pid_b * stride_o_b + pid_n * stride_o_n +
                        offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d)
    tl.store(out_ptrs, output.to(Out_ptr.dtype.element_ty), mask=query_mask)


def mha_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    B, N_CTX, H, D_HEAD = q.shape
    
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
    B, N_CTX, H, D_HEAD = q.shape
    # Use float32 for reference calculation to match CUDA's behavior
    q_b = q.view(B * N_CTX, H, D_HEAD).to(torch.float32)
    k_b = k.view(B * N_CTX, H, D_HEAD).to(torch.float32)
    v_b = v.view(B * N_CTX, H, D_HEAD).to(torch.float32)
    
    scores = torch.bmm(q_b, k_b.transpose(1, 2))
    p = torch.nn.functional.softmax(scores / (D_HEAD**0.5), dim=-1)
    output_b = torch.bmm(p, v_b)
    
    # Cast back to original dtype
    return output_b.view(B, N_CTX, H, D_HEAD).to(q.dtype)


if __name__ == "__main__":
    B, N_CTX, H, D_HEAD = 4, 2048, 6, 256
    # Use float32 as the base type, as in the CUDA kernel
    dtype = torch.float32
    q = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=dtype)
    k = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=dtype)
    v = torch.randn((B, N_CTX, H, D_HEAD), device='cuda', dtype=dtype)

    output_triton = mha_triton(q, k, v)
    output_pytorch = mha_pytorch_reference(q, k, v)

    print(f"Triton implementation running on: {output_triton.device}")
    is_close = torch.allclose(output_triton, output_pytorch, atol=1e-4, rtol=1e-4)
    print(f"Results are close: {is_close}")
    max_diff = (output_triton - output_pytorch).abs().max().item()
    print(f"Maximum absolute difference: {max_diff:.6f}")
    rel_max_diff = (output_triton - output_pytorch).abs().max().item() / output_pytorch.abs().max().item()
    print(f"Relative maximum absolute difference: {rel_max_diff:.6f}")
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Triton implementation running on: cuda:0
Results are close: False
Maximum absolute difference: 2.418357
'''