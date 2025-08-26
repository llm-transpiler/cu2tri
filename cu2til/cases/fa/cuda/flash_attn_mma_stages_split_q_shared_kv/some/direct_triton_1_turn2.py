import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    num_heads, seq_len,
    scale,
    # --- Triton 模板元参数 ---
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton 内核，实现了 FlashAttention。
    - BLOCK_M: Q 序列长度的处理块大小 (对应 CUDA 的 Br)
    - BLOCK_N: K/V 序列长度的处理块大小 (对应 CUDA 的 Bc)
    - BLOCK_DMODEL: 头的维度 (对应 CUDA 的 kHeadDim)
    """
    # 1. 获取程序/线程块的 ID
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # 2. 计算内存偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    offs_o = pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # 3. 初始化
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    # 4. 加载 Q 块
    mask_q = offs_m[:, None] < seq_len
    q = tl.load(Q + offs_q, mask=mask_q, other=0.0)
    
    # 5. 主循环：迭代 K 和 V 的块
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # -- a. 加载 K 和 V 的当前块 --
        
        # K 的偏移量和加载
        # 为了高效计算 Q @ K^T (tl.dot(q, k, trans_b=True))
        # 我们需要加载一个形状为 (BLOCK_DMODEL, BLOCK_N) 的 k 块。
        offs_k = pid_bh * stride_kh + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        mask_k = offs_n[None, :] < seq_len
        k = tl.load(K + offs_k, mask=mask_k, other=0.0)
        
        # V 的偏移量和加载
        # 为了高效计算 P @ V (tl.dot(p_ij, v))
        # 我们需要加载一个形状为 (BLOCK_N, BLOCK_DMODEL) 的 v 块。
        # *** 这是修复之处 ***
        offs_v = pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        mask_v = offs_n[:, None] < seq_len # mask_v 的形状需要与 v 匹配
        v = tl.load(V + offs_v, mask=mask_v, other=0.0)

        # -- b. 计算 S_ij = Q @ K^T --
        s_ij = tl.dot(q, k, trans_b=True) * scale
        
        # -- c. 在线 Softmax 更新 --
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(s_ij - m_new[:, None])
        l_ij = tl.sum(beta, axis=1)
        l_new = alpha * l_i + l_ij
        
        acc *= alpha[:, None]
        p_ij = beta.to(Q.dtype.element_ty)
        acc += tl.dot(p_ij, v)
        
        m_i = m_new
        l_i = l_new

    # 6. 最终归一化和存储
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc /= l_i_safe[:, None]

    tl.store(O + offs_o, acc.to(O.dtype.element_ty), mask=mask_q)


def flash_attn_triton(Q, K, V):
    """
    启动器函数，用于配置和运行 Triton 内核。
    """
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    assert Q.dtype == K.dtype == V.dtype
    assert Q.device == K.device == V.device
    
    batch_size, num_heads, seq_len, d_model = Q.shape
    
    O = torch.empty_like(Q)
    scale = 1.0 / (d_model ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 32
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

    _flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        num_heads, seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=d_model,
        num_warps=4,
        num_stages=2 # 对应 CUDA 内核的 kStage=2
    )
    
    return O


if __name__ == '__main__':
    # 测试代码
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 12, 1024, 64

    assert SEQ_LEN % 64 == 0
    assert SEQ_LEN % 32 == 0

    Q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
    K = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
    V = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')

    print("Running Triton implementation...")
    triton_output = flash_attn_triton(Q, K, V)

    print("Running PyTorch eager implementation for verification...")
    # 使用 PyTorch 2.0+ 的 scaled_dot_product_attention 作为黄金参考
    pytorch_output_pt = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=False
    )

    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output_pt.shape}")
    
    is_close = torch.allclose(triton_output, pytorch_output_pt, atol=1e-2, rtol=0)
    print(f"Outputs are close: {is_close}")

    diff = torch.abs(triton_output - pytorch_output_pt)
    print(f"Max absolute difference: {diff.max().item()}")

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_2.py
Running Triton implementation...
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
TypeError: dot() got an unexpected keyword argument 'trans_b'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_2.py", line 138, in <module>
    triton_output = flash_attn_triton(Q, K, V)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_2.py", line 108, in flash_attn_triton
    _flash_attention_kernel[grid](
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
triton.compiler.errors.CompilationError: at 61:15:
        k = tl.load(K + offs_k, mask=mask_k, other=0.0)

        # V 的偏移量和加载
        # 为了高效计算 P @ V (tl.dot(p_ij, v))
        # 我们需要加载一个形状为 (BLOCK_N, BLOCK_DMODEL) 的 v 块。
        # *** 这是修复之处 ***
        offs_v = pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        mask_v = offs_n[:, None] < seq_len # mask_v 的形状需要与 v 匹配
        v = tl.load(V + offs_v, mask=mask_v, other=0.0)

        # -- b. 计算 S_ij = Q @ K^T --
        s_ij = tl.dot(q, k, trans_b=True) * scale
'''