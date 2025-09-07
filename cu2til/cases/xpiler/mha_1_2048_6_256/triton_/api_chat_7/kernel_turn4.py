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

    # --- THIS IS THE FIX ---
    # The expression (HEAD_DIM**-0.5) is a Python float.
    # Triton automatically handles the type promotion during multiplication.
    scaling_factor = HEAD_DIM ** -0.5
    s = s * scaling_factor
    # -----------------------

    s_mask = (offs_h[:, None] < NUM_HEADS) & (offs_h[None, :] < NUM_HEADS)
    s = tl.where(s_mask, s, -float('inf'))
    
    p = tl.softmax(s, axis=1)

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
    BATCH_SIZE = 1
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    torch.manual_seed(0)
    q_flat = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM), device='cuda', dtype=torch.float32)
    k_flat = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM), device='cuda', dtype=torch.float32)
    v_flat = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM), device='cuda', dtype=torch.float32)
    
    q = q_flat.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    k = k_flat.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    v = v_flat.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)

    triton_output = triton_attention(q, k, v)

    def torch_reference(q, k, v):
        output = torch.zeros_like(q)
        scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
        for i in range(BATCH_SIZE):
            for j in range(SEQ_LEN):
                q_ij = q[i, j, :, :]
                k_ij = k[i, j, :, :]
                v_ij = v[i, j, :, :]
                
                score = torch.matmul(q_ij, k_ij.transpose(-1, -2))
                score = score * scaling_factor
                p = torch.nn.functional.softmax(score, dim=-1)
                
                out_ij = torch.matmul(p, v_ij)
                output[i, j, :, :] = out_ij
        return output

    torch_output = torch_reference(q, k, v)

    print("Triton and PyTorch outputs are close:", torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-4))
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn4.py
Traceback (most recent call last):
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn4.py", line 102, in <module>
    triton_output = triton_attention(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_7/kernel_turn4.py", line 73, in triton_attention
    attention_kernel[grid](
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
triton.compiler.errors.CompilationError: at 41:8:

    # --- THIS IS THE FIX ---
    # The expression (HEAD_DIM**-0.5) is a Python float.
    # Triton automatically handles the type promotion during multiplication.
    scaling_factor = HEAD_DIM ** -0.5
    s = s * scaling_factor
    # -----------------------

    s_mask = (offs_h[:, None] < NUM_HEADS) & (offs_h[None, :] < NUM_HEADS)
    s = tl.where(s_mask, s, -float('inf'))

    p = tl.softmax(s, axis=1)
        ^
TypeError("softmax() got an unexpected keyword argument 'axis'")
'''