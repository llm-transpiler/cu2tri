import torch
import triton
import triton.language as tl

@triton.jit
def mha_triton_kernel_final(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride variables
    stride_b, stride_s, stride_h, stride_d,
    # Constants
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    q_token_ptr = Q_ptr + pid_b * stride_b + pid_s * stride_s
    k_token_ptr = K_ptr + pid_b * stride_b + pid_s * stride_s
    v_token_ptr = V_ptr + pid_b * stride_b + pid_s * stride_s

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    h_mask = offs_h < NUM_HEADS

    q_ptrs = q_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    k_ptrs = k_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    v_ptrs = v_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d

    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=h_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=h_mask[:, None], other=0.0)

    # Manual implementation for Q @ K.T
    q_exp = tl.expand_dims(q, 1)
    k_exp = tl.expand_dims(k, 0)
    scores = tl.sum(q_exp * k_exp, axis=2)

    # --- THE FIX IS ON THE NEXT LINE ---
    # Changed HEAD_DIM.to(tl.float32) to float(HEAD_DIM)
    scaling_factor = 1.0 / tl.sqrt(float(HEAD_DIM))

    scores = tl.where(h_mask[:, None] & h_mask[None, :], scores, float("-inf"))
    scores *= scaling_factor
    attn_probs = tl.softmax(scores, axis=1) # Corrected softmax axis for row-wise
    attn_probs = attn_probs.to(tl.float32)

    # Manual implementation for Scores @ V
    p_exp = tl.expand_dims(attn_probs, 2)
    v_exp = tl.expand_dims(v, 0)
    output_block = tl.sum(p_exp * v_exp, axis=1)

    o_token_ptr = O_ptr + pid_b * stride_b + pid_s * stride_s
    o_ptrs = o_token_ptr + offs_h[:, None] * stride_h + offs_d[None, :] * stride_d
    tl.store(o_ptrs, output_block, mask=h_mask[:, None])

# The Python wrapper remains the same
def mha_triton_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    output = torch.empty_like(q)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    grid = (batch_size, seq_len)
    BLOCK_H = triton.next_power_of_2(num_heads)

    mha_triton_kernel_final[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=BLOCK_H,
    )
    return output
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/test_torch_triton.py
Multi-Head Attention PyTorch vs Triton 对比测试
=======================================================
✅ Triton kernel加载成功
✅ GPU可用: NVIDIA RTX 6000 Ada Generation

=======================================================
测试1: 简单测试用例 (快速验证)
=======================================================
创建简单测试: batch=1, seq_len=32, heads=4, dim=64
输入形状: torch.Size([1, 32, 4, 64])

运行简单测试...
PyTorch完成: mean=0.000013, std=0.049960
Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/test_torch_triton.py", line 278, in <module>
    main()
  File "/workspace/cu2til/cases/mha/test_torch_triton.py", line 213, in main
    output_triton_simple = triton_mha(Q_simple, K_simple, V_simple, triton_kernel)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/test_torch_triton.py", line 85, in triton_mha
    output = triton_kernel(Q, K, V)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 66, in mha_triton_wrapper
    mha_triton_kernel_final[grid](
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
triton.compiler.errors.CompilationError: at 41:17:
    # Manual implementation for Q @ K.T
    q_exp = tl.expand_dims(q, 1)
    k_exp = tl.expand_dims(k, 0)
    scores = tl.sum(q_exp * k_exp, axis=2)

    # --- THE FIX IS ON THE NEXT LINE ---
    # Changed HEAD_DIM.to(tl.float32) to float(HEAD_DIM)
    scaling_factor = 1.0 / tl.sqrt(float(HEAD_DIM))

    scores = tl.where(h_mask[:, None] & h_mask[None, :], scores, float("-inf"))
    scores *= scaling_factor
    attn_probs = tl.softmax(scores, axis=1) # Corrected softmax axis for row-wise
                 ^
TypeError("softmax() got an unexpected keyword argument 'axis'")
'''