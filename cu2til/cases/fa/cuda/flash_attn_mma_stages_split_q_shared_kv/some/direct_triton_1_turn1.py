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
    # 对应 CUDA 内核中的 Br, Bc, kHeadDim
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
    # 每个程序实例处理一个 Q 块
    pid_m = tl.program_id(0)
    # 每个程序实例处理一个 batch 和 head
    pid_bh = tl.program_id(1)

    # 2. 计算内存偏移量
    # 计算当前块的 Q, K, V, O 的起始偏移量
    # Q 的偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    
    # K 和 V 的偏移量将在循环中计算
    # O 的偏移量
    offs_o = pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # 3. 初始化
    # 累加器，用于存储 O 的结果 (P @ V)
    # 使用 float32 以获得更高的精度
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    # 在线 softmax 的统计量
    # l_i: 行指数和的累加 (row-wise sum of exp)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # m_i: 行最大值的累加 (row-wise max)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    # 4. 加载 Q 块
    # Q 块在整个内层循环中是常量，所以只加载一次
    # 创建掩码以处理序列末尾的填充
    mask_q = offs_m[:, None] < seq_len
    q = tl.load(Q + offs_q, mask=mask_q, other=0.0)
    
    # 5. 主循环：迭代 K 和 V 的块
    # 这对应于 CUDA 内核中的 `for (int tile_K_seqlen = 0; ...)`
    # Triton 编译器会自动处理这里的流水线（pipelining），
    # 类似于 CUDA 内核中的 `kStage=2` 和 `cp.async`。
    for start_n in range(0, seq_len, BLOCK_N):
        # -- a. 加载 K 和 V 的当前块 --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # K 的偏移量和加载
        offs_k = pid_bh * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        mask_k = offs_n[None, :] < seq_len
        k = tl.load(K + offs_k, mask=mask_k, other=0.0)
        
        # V 的偏移量和加载
        offs_v = pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[:, None] * stride_vk
        # mask_v 和 mask_k 相同
        v = tl.load(V + offs_v, mask=mask_k, other=0.0)

        # -- b. 计算 S_ij = Q @ K^T --
        # s_ij 是一个 (BLOCK_M, BLOCK_N) 的块
        s_ij = tl.dot(q, k, trans_b=True) * scale
        # 应用掩码，防止注意力机制看到未来的 token 或填充
        # 对于自回归模型，需要一个因果掩码
        # causal_mask = offs_m[:, None] >= offs_n[None, :]
        # s_ij = tl.where(causal_mask, s_ij, -float("inf"))
        # 这个内核实现的是双向注意力，所以不需要因果掩码

        # -- c. 在线 Softmax 更新 --
        # 1. 找到当前块的最大值
        m_ij = tl.max(s_ij, axis=1)
        
        # 2. 更新全局最大值
        m_new = tl.maximum(m_i, m_ij)
        
        # 3. 计算缩放因子和当前块的 softmax
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(s_ij - m_new[:, None])
        
        # 4. 更新行指数和
        l_ij = tl.sum(beta, axis=1)
        l_new = alpha * l_i + l_ij
        
        # 5. 更新输出累加器 `acc`
        # `acc` 需要用旧的缩放因子进行缩放
        acc *= alpha[:, None]
        
        # 计算 P_ij @ V_j 并加到 `acc`
        # 将 beta 转换回 Q 的数据类型以进行点积
        p_ij = beta.to(Q.dtype.element_ty)
        acc += tl.dot(p_ij, v)
        
        # 6. 为下一次迭代更新统计量
        m_i = m_new
        l_i = l_new

    # 6. 最终归一化和存储
    # 循环结束后，`acc` 包含了 O 的非归一化值
    # 用最终的行指数和 `l_i` 进行归一化
    # 加上一个小的 epsilon 防止除以零
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc /= l_i_safe[:, None]

    # 将最终结果写回全局内存
    tl.store(O + offs_o, acc.to(O.dtype.element_ty), mask=mask_q)


def flash_attn_triton(Q, K, V):
    """
    启动器函数，用于配置和运行 Triton 内核。
    """
    # 检查输入张量
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    assert Q.dtype == K.dtype == V.dtype
    assert Q.device == K.device == V.device
    
    batch_size, num_heads, seq_len, d_model = Q.shape
    
    # 创建输出张量
    O = torch.empty_like(Q)
    
    # 定义缩放因子
    scale = 1.0 / (d_model ** 0.5)

    # 定义 Triton 内核的块大小
    # 这些值应该根据 GPU 架构和 d_model 进行调整以获得最佳性能
    # 这里我们选择与 CUDA 内核匹配的值
    BLOCK_M = 64
    BLOCK_N = 32
    
    # 定义网格大小
    # grid_x: 沿 Q 序列长度的块数
    # grid_y: batch * num_heads
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

    # 运行内核
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
        # num_warps 和 num_stages 可以用来微调性能
        # num_warps=4,
        # num_stages=2 # 对应 CUDA 内核的 kStage=2
    )
    
    return O

# --- Pybind11 接口和测试 ---
# 假设这是在一个 Python 文件中，可以直接调用
# 如果需要编译成 .so，则需要类似原始代码的 Pybind11 封装

def flash_attn_mma_stages_split_q_shared_kv(Q, K, V, O_ref):
    """
    一个包装函数，使其接口与原始 C++ 函数的 Pybind 包装器匹配。
    注意：Triton 版本通常返回一个新的张量，而不是修改一个预先分配的张量。
    为了匹配接口，我们可以将结果写入 O_ref。
    """
    O_triton = flash_attn_triton(Q, K, V)
    # 这里的 O_ref 实际上没有被使用，因为 Triton 版本会自己创建输出
    # 在实际使用中，我们会直接返回 O_triton
    print("Triton implementation ran. Returning the result.")
    # 如果非要匹配 C++ 的输出参数模式，可以这样做：
    # O_ref.copy_(O_triton)
    # 但这会引入一次额外的拷贝。
    return O_triton


if __name__ == '__main__':
    # 测试代码
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 12, 1024, 64

    # 确保 SEQ_LEN 是块大小的倍数以简化测试
    assert SEQ_LEN % 64 == 0
    assert SEQ_LEN % 32 == 0

    # 创建输入数据
    Q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
    K = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')
    V = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), dtype=torch.float16, device='cuda')

    # 运行 Triton 版本
    print("Running Triton implementation...")
    triton_output = flash_attn_triton(Q, K, V)

    # 运行 PyTorch eager 模式作为参考
    print("Running PyTorch eager implementation for verification...")
    scale = 1.0 / (D_HEAD ** 0.5)
    # PyTorch 的 sdpa 不支持 4D 输入，需要 reshape
    Q_pt = Q.transpose(1, 2).reshape(BATCH * SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 1)
    K_pt = K.transpose(1, 2).reshape(BATCH * SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 1)
    V_pt = V.transpose(1, 2).reshape(BATCH * SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 1)
    
    # 使用 PyTorch 2.0 的 scaled_dot_product_attention
    pytorch_output_pt = torch.nn.functional.scaled_dot_product_attention(
        Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2), is_causal=False
    ).transpose(1,2)

    # 比较结果
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output_pt.shape}")
    
    # 检查数值差异
    # 由于在线 softmax 的计算顺序和浮点数精度问题，结果可能存在微小差异
    is_close = torch.allclose(triton_output, pytorch_output_pt, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_close}")

    # 打印一些差异统计
    diff = torch.abs(triton_output - pytorch_output_pt)
    print(f"Max absolute difference: {diff.max().item()}")

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1.py
Running Triton implementation...
Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1.py", line 207, in <module>
    triton_output = flash_attn_triton(Q, K, V)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/direct_triton_1.py", line 154, in flash_attn_triton
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
triton.compiler.errors.CompilationError: at 68:17:
    # 类似于 CUDA 内核中的 `kStage=2` 和 `cp.async`。
    for start_n in range(0, seq_len, BLOCK_N):
        # -- a. 加载 K 和 V 的当前块 --
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K 的偏移量和加载
        offs_k = pid_bh * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        mask_k = offs_n[None, :] < seq_len
        k = tl.load(K + offs_k, mask=mask_k, other=0.0)

        # V 的偏移量和加载
        offs_v = pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[:, None] * stride_vk
                 ^
ValueError('Cannot make_shape_compatible: incompatible dimensions at index 0: 32 and 64')
'''