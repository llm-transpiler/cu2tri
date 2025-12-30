import torch
import triton
import triton.language as tl
import math

# The Triton JIT-compiled kernel
@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_h = offs_h < NUM_HEADS

    qkv_offs = offs_h[:, None] * stride_q_h + offs_d[None, :]
    qkv_mask = mask_h[:, None]

    q_ptr = Q + batch_idx * stride_q_b + seq_idx * stride_q_s
    k_ptr = K + batch_idx * stride_k_b + seq_idx * stride_k_s
    v_ptr = V + batch_idx * stride_v_b + seq_idx * stride_v_s

    q = tl.load(q_ptr + qkv_offs, mask=qkv_mask, other=0.0)
    k = tl.load(k_ptr + qkv_offs, mask=qkv_mask, other=0.0)
    v = tl.load(v_ptr + qkv_offs, mask=qkv_mask, other=0.0)

    s = tl.dot(q, tl.trans(k))

    scaling_factor = HEAD_DIM ** -0.5
    s = s * scaling_factor

    s_mask = (offs_h[:, None] < NUM_HEADS) & (offs_h[None, :] < NUM_HEADS)
    s = tl.where(s_mask, s, -float('inf'))
    
    # --- THIS IS THE FIX ---
    # The 'axis' argument is not present in older Triton versions.
    # The default behavior is to compute softmax over the last dimension, which is what we want.
    p = tl.softmax(s)
    # -----------------------

    p = p.to(v.dtype)
    output = tl.dot(p, v)

    out_ptr = Out + batch_idx * stride_o_b + seq_idx * stride_o_s
    out_offs = offs_h[:, None] * stride_o_h + offs_d[None, :]
    tl.store(out_ptr + out_offs, output, mask=qkv_mask)


# Python wrapper function (no changes needed)
def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    output = torch.empty_like(q)
    
    grid = (batch_size, seq_len)
    
    block_h = max(16, triton.next_power_of_2(num_heads))

    attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
    )
    
    return output

# --- Verification ---
if __name__ == '__main__':
    batch_size = 1
    seq_len = 2048
    num_heads = 6
    head_dim = 256

    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    k = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    v = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)

    triton_output = triton_attention(q, k, v)

    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    p = torch.nn.functional.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(p, v)
    
    print("Triton vs. PyTorch allclose:", torch.allclose(triton_output, pytorch_output, atol=1e-5, rtol=1e-4))
    
    print("\nTriton output (slice):")
    print(triton_output[0, 0, :3, :5])
    
    print("\nPyTorch output (slice):")
    print(pytorch_output[0, 0, :3, :5])
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn5_debug.py
Triton vs. PyTorch allclose: False

Triton output (slice):
tensor([[nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan]], device='cuda:0')

PyTorch output (slice):
tensor([[-0.1168, -0.5989, -0.4364,  0.4442, -0.1245],
        [-1.1084, -0.2109, -0.6088,  0.8212, -0.5519],
        [-0.2770, -0.0590, -0.6516,  0.4019, -0.2279]], device='cuda:0')
'''