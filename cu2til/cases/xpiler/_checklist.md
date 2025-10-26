sumpool_4_8_8_64_3_3_2_2
2309->2304
softmax_5_12_23_128
threadIdx.x -> idx

GQA 类（_xpiler_src/gqa_*_16_16_512）修复说明（已完成 ✅）
- 适用范围：`gqa_1_2_16_16_512/ gqa_1_4_16_16_512/ gqa_2_2_16_16_512/ gqa_2_4_16_16_512/ gqa_4_2_16_16_512/ gqa_4_4_16_16_512/ gqa_8_2_16_16_512/ gqa_8_4_16_16_512`
- 核心修改（相较于 ref.cu）：
  1. **真正的 GQA 实现**：
     - kernel 签名新增 `int num_q_heads, int num_kv_heads` 参数
     - 实现正确的分组逻辑：`int q_head = blockIdx.z % num_q_heads; int kv_head = q_head / (num_q_heads / num_kv_heads);`
     - Q 数据：`[batch*num_q_heads, SEQ_Q, HEAD_DIM]`，K/V 数据：`[batch*num_kv_heads, SEQ_KV, HEAD_DIM]`
     - 每组 Q heads 共享一个 KV head，实现真正的 GQA（而非 MHA）
  2. **数值精度优化**：
     - `attn_tile` 使用 `float[BLOCK_M * SEQ_KV]` 代替 `half[]`，避免 softmax 中间值精度损失
     - softmax 计算全程使用 float 精度（max/exp/sum/归一化）
     - 最终概率转为 half 存入 `softmax_tile`
  3. **正确的 attention scaling**：
     - 应用标准缩放因子 `scale = 1.0f / sqrtf((float)HEAD_DIM)`
     - softmax 前对 scores 乘以 scale，与标准 attention 一致
  4. **网格划分与地址计算**：
     - 网格：`dim3 grid((M+BLOCK_M-1)/BLOCK_M, 1, batch*num_q_heads)` - 每个 Q head 独立处理
     - batch/head 解耦：`int batch = blockIdx.z / num_q_heads; int q_head = blockIdx.z % num_q_heads;`
     - 指针计算考虑 GQA 布局：`q_ptr = Q + (batch*num_q_heads + q_head)*SEQ_Q*HEAD_DIM + row0*HEAD_DIM`
     - K/V 共享：`k_ptr/v_ptr = K/V + (batch*num_kv_heads + kv_head)*SEQ_KV*HEAD_DIM`
  5. **WMMA 载入**：
     - Q: row_major, ld=HEAD_DIM
     - K: col_major, ld=HEAD_DIM（计算 Q@K^T）
     - V: row_major, ld=HEAD_DIM
     - 所有 accumulator 使用 float 类型
  6. **代码精简**：
     - Fragment 声明单行化
     - 使用 `min()` 计算 `rows_valid`
     - 复用 attn_tile 作为输出临时空间
- Torch 参考（torch_/ref.py）：
  - 实现完整 GQA 逻辑：`group_size = num_q_heads // num_kv_heads; kv_idx = q_idx // group_size`
  - 每个 Q head 选择对应的 KV head 进行 attention 计算
  - 使用 float32 计算，应用 `1/sqrt(d)` 缩放
  - 最终拼接所有 Q head 的输出
- 数据构造（get_data.py）：
  - Q: `[batch*num_q_heads, M, K_dim]` - 例如 `[2, 16, 16]`
  - K/V: `[batch*num_kv_heads, N, K_dim]` - 例如 `[1, 512, 16]`
  - 半精度（fp16），使用 normal 分布初始化
  - CUDA 参数：`Q, K, V, O, batch, num_q_heads, num_kv_heads, M, K_dim, N`
- 测试结果：
  - 所有配置测试通过（Max Abs Err < 3.05e-05）
  - gqa_4_4 (16 Q heads, 4 KV heads) 相比 PyTorch 达到 2.5x 加速

GEMM 类（xpiler/gemm_*/）修复说明（统一）
- 适用范围：`gemm_32_32_128/ gemm_32_32_1024/ gemm_32_128_128/ gemm_32_128_1024/ gemm_1024_16_128/ gemm_1024_16_1024/ gemm_1024_128_128/ gemm_1024_128_4096`
- 修复要点：
  - B 的布局与步幅与内存一致：若内存以 `B + k*n + blockCol` 方式访问，则 WMMA 载入使用 `row_major` + `ld = n`；若以 `B + blockCol*ld + k` 方式访问，则为 `col_major` + `ld = ld`。
  - 网格：`dim3 numBlocks((n+16-1)/16, (m+16-1)/16)`，每个 block 负责 16×16 输出块；K 维内循环 `k += 16`。
  - 累加：`wmma::accumulator` 用 float；`fill_fragment(c_frag, 0.0f)`。
  - 存储：`store_matrix_sync(C + blockRow * n + blockCol, c_frag, n, row_major)`。
  - 与参考差异的统一修复：确保 `matrix_b` 的 `row_major/col_major` 与 `load_matrix_sync` 的 `ld` 匹配实际内存步幅，避免错步幅导致数值错误。

注：`ref.cu` 若展示了 `(M,N)` 双向切分但未在 kernel 内真正使用 `grid.y`，属于外壳不一致；正确实现应避免 softmax 跨 block 归约，或采用双核归约方案。