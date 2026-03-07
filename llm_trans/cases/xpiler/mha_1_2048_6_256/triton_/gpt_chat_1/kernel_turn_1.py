import triton
import triton.language as tl

# Q, K, V, O 都被视作 float32 连续内存，其逻辑形状为 [B, S, H, D]
# 索引规则与原 CUDA 内核一致：i*S*H*D + j*H*D + h*D + d
@triton.jit
def mix_heads_kernel(
    Q, K, V, O,
    stride_i, stride_j, stride_h, stride_d,      # 各维的步长（元素为单位）
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)        # 1D grid：每个 program 处理一个 (i, j)
    i = pid // S                  # batch index
    j = pid % S                   # seq index (query index)

    hs = tl.arange(0, H)          # [H]
    ds = tl.arange(0, D)          # [D]

    # 组装指针，形成 [H, D] 的 tile
    q_ptrs = Q + i*stride_i + j*stride_j + hs[:, None]*stride_h + ds[None, :]*stride_d
    k_ptrs = K + i*stride_i + j*stride_j + hs[:, None]*stride_h + ds[None, :]*stride_d
    v_ptrs = V + i*stride_i + j*stride_j + hs[:, None]*stride_h + ds[None, :]*stride_d
    o_ptrs = O + i*stride_i + j*stride_j + hs[:, None]*stride_h + ds[None, :]*stride_d

    # 加载 [H, D]
    q = tl.load(q_ptrs).to(tl.float32)
    k = tl.load(k_ptrs).to(tl.float32)
    v = tl.load(v_ptrs).to(tl.float32)

    # score = Q @ K^T   -> [H, H]
    scores = tl.dot(q, tl.trans(k))

    # 缩放（与原代码相同：1/sqrt(256)）
    scores = scores * (1.0 / (D ** 0.5))

    # softmax 沿着列维（每个 m 对 n 做 softmax），不做减最大值以完全复现原始数值行为
    exp_scores = tl.exp(scores)           # [H, H]
    denom = tl.sum(exp_scores, axis=1)    # [H]
    probs = exp_scores / denom[:, None]   # [H, H]

    # 输出：out = probs @ V   -> [H, D]
    out = tl.dot(probs, v)                # [H, D]
    tl.store(o_ptrs, out)


def cuda_kernel_triton(d_queries, d_keys, d_values, d_output,
                       batch_size=1, seq_len=2048, num_heads=6, head_dim=256,
                       stream=None):
    """
    d_queries / d_keys / d_values / d_output: torch.cuda.FloatTensor 或 Triton 支持的设备指针
    逻辑形状: [B, S, H, D]，物理上最后一维连续（C 连续）
    """
    # 与原 CUDA 的索引公式一致的步长（以元素为单位）
    stride_i = seq_len * num_heads * head_dim    # S*H*D
    stride_j = num_heads * head_dim              # H*D
    stride_h = head_dim                          # D
    stride_d = 1

    # 1D grid：每个 program 处理一个 (i, j)
    grid = (batch_size * seq_len,)

    mix_heads_kernel[grid](
        d_queries, d_keys, d_values, d_output,
        stride_i, stride_j, stride_h, stride_d,
        S=seq_len, H=num_heads, D=head_dim,
        num_warps=1, num_stages=1,                # 对应这个小 tile 已足够；可按需求调整
        stream=stream
    )

