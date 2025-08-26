
import torch
import triton
import triton.language as tl
# @triton.autotune(
#     configs=[
#         # 定义一系列要测试的配置
#         triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
#         for BM in [64, 128]
#         for BN in [32, 64]
#         for s in [2, 3, 4, 7]
#         for w in [4, 8]
#     ],
#     # 定义用于缓存编译结果的 key。
#     key=['seq_len', 'd_model'],
# )
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
        # 为了计算 Q @ K^T，我们需要加载一个 K 块，转置后形状为 (DMODEL, N)。
        # 因此，我们加载一个形状为 (N, DMODEL) 的块。
        # *** 这是修复之处 1: 改变 k 的加载形状 ***
        offs_k = pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        mask_k = offs_n[:, None] < seq_len
        k = tl.load(K + offs_k, mask=mask_k, other=0.0)
        
        # V 的偏移量和加载
        # 为了计算 P @ V，我们需要加载一个形状为 (N, DMODEL) 的 V 块。
        offs_v = pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        mask_v = offs_n[:, None] < seq_len
        v = tl.load(V + offs_v, mask=mask_v, other=0.0)

        # -- b. 计算 S_ij = Q @ K^T --
        # *** 这是修复之处 2: 使用 .T 替代 trans_b=True ***
        s_ij = tl.dot(q, k.T) * scale
        
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
        num_stages=2 
    )
    
    return O



def test_non_causal_attention(BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    sm_scale = 1. / (HEAD_DIM ** 0.5)
    
    # 参考实现
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    
    # triton实现
    tri_out = flash_attn_triton(q, k, v)
    # print("max diff:", (ref_out - tri_out).abs().max().item())
    
    # 比较
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print("Non-casual attention pass.")


from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
HAS_FLASH = True

BATCH_SIZE, N_HEADS, HEAD_DIM = 4, 32, 64

# 非因果注意力基准测试配置
configs_non_causal = []
for mode in ["fwd"]:
    configs_non_causal.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"non-causal-attention-batch{BATCH_SIZE}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
            args={
                "HEAD_NUM": N_HEADS,
                "BATCH_SIZE": BATCH_SIZE,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
            },
        ))

@triton.testing.perf_report(configs_non_causal)
def bench_non_causal_attention(BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM, mode, provider, device="cuda"):
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
        k = torch.randn((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
        v = torch.randn((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
        # sm_scale = 1.3
        fn = lambda: flash_attn_triton(q, k, v)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        qkv = torch.randn((BATCH_SIZE, SEQ_LEN, 3, HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)
        fn = lambda: flash_attn_func(qkv, causal=False)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH_SIZE * HEAD_NUM * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # 运行测试
    print("开始测试...")
    test_non_causal_attention(4, 32, 1024, 64)
    print("测试完成！")
    
    # 运行基准测试
    print("开始基准测试...")
    bench_non_causal_attention.run(save_path=".", print_data=True)
    print("基准测试完成！")

# if __name__ == '__main__':
#     # 测试代码
#     BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 12, 1024, 64

#     assert SEQ_LEN % 64 == 0
#     assert SEQ_LEN % 32 == 0

#     Q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
#     K = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
#     V = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')

#     print("Running Triton implementation...")
#     triton_output = flash_attn_triton(Q, K, V)

#     print("Running PyTorch eager implementation for verification...")
#     # 使用 PyTorch 2.0+ 的 scaled_dot_product_attention 作为黄金参考
#     pytorch_output_pt = torch.nn.functional.scaled_dot_product_attention(
#         Q, K, V, is_causal=False
#     )

#     print(f"Triton output shape: {triton_output.shape}")
#     print(f"PyTorch output shape: {pytorch_output_pt.shape}")
    
#     is_close = torch.allclose(triton_output, pytorch_output_pt, atol=1e-2, rtol=0)
#     print(f"Outputs are close: {is_close}")

#     diff = torch.abs(triton_output - pytorch_output_pt)
#     print(f"Max absolute difference: {diff.max().item()}")

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
Running Triton implementation...
Running PyTorch eager implementation for verification...
Triton output shape: torch.Size([4, 12, 1024, 64])
PyTorch output shape: torch.Size([4, 12, 1024, 64])
Outputs are close: True
Max absolute difference: 0.00048828125
'''

'''
fixed config
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     149.808447  142.047303
1   2048.0     137.094775  143.781707
2   4096.0     107.997625  120.792301
3   8192.0     106.965040  118.538103
4  16384.0     109.430100  129.845581
基准测试完成！
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     143.472744  141.089772
1   2048.0     112.999580  120.710733
2   4096.0     105.067474  124.425447
3   8192.0     104.429279  128.496422
4  16384.0     110.697889  129.824000
基准测试完成！
清cache
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     147.067199  141.748097
1   2048.0     107.465773  118.567241
2   4096.0     107.674844  122.997302
3   8192.0     109.285311  132.519817
4  16384.0     109.445433  132.075620
基准测试完成！
'''
'''
@triton.autotune(
    configs=[
        # 定义一系列要测试的配置
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64]
        for s in [3, 4, 7]
        for w in [4, 8]
    ],
    # 定义用于缓存编译结果的 key。
    key=['seq_len', 'd_model'],
)
add config
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     133.874716  151.408641
1   2048.0      94.531072  113.200222
2   4096.0     103.735882  117.228869
3   8192.0     106.786855  113.575401
4  16384.0     103.794567  121.080501
基准测试完成！
'''

'''
@triton.autotune(
    configs=[
        # 定义一系列要测试的配置
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64]
        for s in [2, 3, 4, 7]
        for w in [4, 8]
    ],
    # 定义用于缓存编译结果的 key。
    key=['seq_len', 'd_model'],
)
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     137.024655  140.149667
1   2048.0      94.393143  114.373063
2   4096.0     100.035235  118.413726
3   8192.0     102.112819  111.225937
4  16384.0      99.197803  119.344431

root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     122.374634  134.621170
1   2048.0      90.505745  107.452861
2   4096.0     100.043823  112.111703
3   8192.0      97.190996  109.505429
4  16384.0      99.013712  116.091183
基准测试完成！
'''