import torch
import triton
import triton.language as tl
import math

# Triton Kernel (Final Correct Version)
@triton.jit
def mha_triton_kernel(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    # Stride information
    stride_q_b, stride_q_s, stride_q_h,
    stride_k_b, stride_k_s, stride_k_h,
    stride_v_b, stride_v_s, stride_v_h,
    stride_o_b, stride_o_s, stride_o_h,
    # Compile-time constants
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # 1. Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Define block shapes for loading. Must be powers of 2.
    # We find the next power of 2 for dimensions that are not already.
    BLOCK_H = tl.next_power_of_2(NUM_HEADS)
    BLOCK_D = HEAD_DIM # 256 is already a power of 2

    # 3. Create pointers and masks for loading
    # Pointers to the start of the (batch, seq_pos) slice
    q_start_ptr = Q_ptr + pid_batch * stride_q_b + pid_seq * stride_q_s
    k_start_ptr = K_ptr + pid_batch * stride_k_b + pid_seq * stride_k_s
    v_start_ptr = V_ptr + pid_batch * stride_v_b + pid_seq * stride_v_s

    # Create coordinate ranges for masking
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Create masks to avoid loading out-of-bounds data
    mask_h = offs_h < NUM_HEADS
    mask_d = offs_d < HEAD_DIM
    # 2D mask for the entire block
    load_mask = mask_h[:, None] & mask_d[None, :]

    # Load Q, K, V with masking
    # The `mask` argument prevents out-of-bounds memory access.
    # The `other` argument fills the padded areas with 0.0.
    q = tl.load(q_start_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :], mask=load_mask, other=0.0)
    k = tl.load(k_start_ptr + offs_h[:, None] * stride_k_h + offs_d[None, :], mask=load_mask, other=0.0)
    v = tl.load(v_start_ptr + offs_h[:, None] * stride_v_h + offs_d[None, :], mask=load_mask, other=0.0)

    # 4. Compute Q @ K.T
    # q is [BLOCK_H, BLOCK_D], k is [BLOCK_H, BLOCK_D]
    # Padded rows/cols in q and k are zero, so dot product is correct.
    scores = tl.dot(q, tl.trans(k)) # Result shape: [BLOCK_H, BLOCK_H]

    # 5. Apply scaling and softmax
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    scores *= scaling_factor
    
    # We must mask the scores before softmax.
    # We set the scores for padded tokens to -inf.
    # This ensures they don't contribute to the softmax sum.
    scores_mask = (offs_h[:, None] < NUM_HEADS) & (offs_h[None, :] < NUM_HEADS)
    scores = tl.where(scores_mask, scores, -float('inf'))
    
    attn_weights = tl.softmax(scores, axis=1)
    attn_weights = attn_weights.to(tl.float32)

    # 6. Compute Softmax @ V
    # attn_weights is [BLOCK_H, BLOCK_H], v is [BLOCK_H, BLOCK_D]
    # Padded rows in V are zero, so they don't affect the output.
    output = tl.dot(attn_weights, v) # Result shape: [BLOCK_H, BLOCK_D]

    # 7. Store the result back to global memory
    output_start_ptr = output_ptr + pid_batch * stride_o_b + pid_seq * stride_o_s
    # Use the same `load_mask` to only write to valid memory locations.
    tl.store(output_start_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :], output, mask=load_mask)


# Wrapper and verification code remain the same
def mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    output = torch.empty_like(q)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    grid = (BATCH, SEQ_LEN)

    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        SEQ_LEN=SEQ_LEN,
        NUM_HEADS=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
    )
    return output

def torch_reference_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    output = torch.zeros_like(q)
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    
    for b in range(BATCH):
        for s in range(SEQ_LEN):
            q_slice = q[b, s, :, :]
            k_slice = k[b, s, :, :]
            v_slice = v[b, s, :, :]
            
            scores = torch.matmul(q_slice, k_slice.T)
            scores *= scaling_factor
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            output_slice = torch.matmul(attn_weights, v_slice)
            output[b, s, :, :] = output_slice
            
    return output

if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)

    triton_output = mha(q, k, v)

    print("Running verification on a smaller tensor size to avoid long wait times...")
    q_small = q[:, :4, :, :].clone()
    k_small = k[:, :4, :, :].clone()
    v_small = v[:, :4, :, :].clone()
    
    triton_output_small = mha(q_small, k_small, v_small)
    torch_output_small = torch_reference_mha(q_small, k_small, v_small)

    print(f"Triton output shape: {triton_output.shape}")
    if torch.allclose(triton_output_small, torch_output_small, atol=1e-5, rtol=1e-4):
        print("✅ Triton and PyTorch results match!")
    else:
        print("❌ Triton and PyTorch results DO NOT match!")
        print("Max difference:", (triton_output_small - torch_output_small).abs().max().item())
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 131, in <module>
    triton_output = mha(q, k, v)
                    ^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 90, in mha
    mha_triton_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 390, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 281, in compile
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
                            ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 78, in hash
    key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"
             ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 696, in cache_key
    dependencies_finder.visit(self.parse())
  File "/usr/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 163, in visit_FunctionDef
    self.generic_visit(node)
  File "/usr/lib/python3.12/ast.py", line 415, in generic_visit
    self.visit(item)
  File "/usr/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 217, in visit_Assign
    self.generic_visit(node)
  File "/usr/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/lib/python3.12/ast.py", line 417, in generic_visit
    self.visit(value)
  File "/usr/lib/python3.12/ast.py", line 407, in visit
    return visitor(node)
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 156, in visit_Attribute
    ret = getattr(lhs, node.attr)
          ^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'triton.language' has no attribute 'next_power_of_2'
'''