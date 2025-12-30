# same as /workspace/cu2til/cases/fa/check/triton_non_causal.py
"""
融合注意力
===============

这是Flash Attention v2算法的Triton实现，来自Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

致谢: OpenAI内核团队

额外致谢:

* 原始flash attention论文 (https://arxiv.org/abs/2205.14135)
* Rabe和Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""
import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "Ada"

import pytest
import torch

import triton
import triton.language as tl


# 我们不每次都运行自动调优以保持教程快速。保留
# 下面的代码并注释掉等效参数便于重新调优。
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

# 非因果注意力kernel
@triton.autotune(list(filter(keep, configs)), key=["SEQ_LEN", "HEAD_DIM"])
@triton.jit
def _attn_fwd_non_causal(Q, K, V, sm_scale, M, Out,  #
                         stride_qz, stride_qh, stride_qm, stride_qk,  #
                         stride_kz, stride_kh, stride_kn, stride_kk,  #
                         stride_vz, stride_vh, stride_vk, stride_vn,  #
                         stride_oz, stride_oh, stride_om, stride_on,  #
                         BATCH_SIZE, HEAD_NUM, SEQ_LEN,  #
                         HEAD_DIM: tl.constexpr,  #
                         BLOCK_M: tl.constexpr,  #
                         BLOCK_N: tl.constexpr  #
                         ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // HEAD_NUM
    off_h = off_hz % HEAD_NUM
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # 块指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # 初始化偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # 初始化指向m和l的指针
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # 加载缩放因子
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # 加载q：它将在整个过程中保留在SRAM中
    q = tl.load(Q_block_ptr)
    
    # 非因果注意力：处理所有位置
    lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    # 循环遍历k, v并更新累加器
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- 计算qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        
        # 不应用掩码，直接计算
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- 更新m_i和l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- 更新输出累加器 --
        acc = acc * alpha[:, None]
        # 更新acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # 更新m_i和l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    # 尾声
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * SEQ_LEN + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))



def attention_non_causal(q, k, v, sm_scale):
    """
    非因果注意力 (Non-Causal Attention)
    """
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)

    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd_non_causal[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        SEQ_LEN=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K)
    
    return o


def test_non_causal_attention(BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((BATCH_SIZE, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    sm_scale = 0.5
    
    # 参考实现
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    
    # triton实现
    tri_out = attention_non_causal(q, k, v, sm_scale)
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
        sm_scale = 1.3
        fn = lambda: attention_non_causal(q, k, v, sm_scale)
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
