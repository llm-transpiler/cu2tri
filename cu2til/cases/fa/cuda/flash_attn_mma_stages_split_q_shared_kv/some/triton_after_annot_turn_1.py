import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton Kernel for Flash Attention

    This kernel is a Triton implementation of the provided CUDA code.
    It performs forward attention calculation in a tiled manner.

    Mapping from CUDA concepts to Triton:
    - Grid (blockIdx.x, blockIdx.y):
        - `blockIdx.x` -> `pid_m`: Iterates over the sequence length of Q.
        - `blockIdx.y` -> `pid_zh`: Iterates over batch and head dimensions.
    - Tiling (Br, Bc, kHeadDim):
        - `Br` (Block Row) -> `BLOCK_SIZE_M`: Tile size for Q's sequence length.
        - `Bc` (Block Column) -> `BLOCK_SIZE_N`: Tile size for K/V's sequence length.
        - `kHeadDim` -> `HEAD_DIM`: The dimension of the attention head.
        - Inner loop over `kHeadDim` -> `BLOCK_SIZE_K`: The inner dimension of the matrix multiplication.
    - Memory Management:
        - `GlobalPtr`, `SharedPtr`, `Register` -> Triton pointers and tensors.
        - `cp.async`, `ldmatrix`, `st.global` -> `tl.load`, `tl.store`. Triton manages shared memory and data movement automatically.
        - Software Pipelining (`kStage=2`) -> Triton's compiler automatically pipelines the main loop over `BLOCK_SIZE_N`.
    - Computation:
        - `mma.sync` -> `tl.dot`. Triton uses Tensor Cores when dtypes and dimensions are appropriate.
        - Element-wise ops, reductions (`warp_shuffle_max`, etc.) -> `tl.max`, `tl.sum`, `tl.exp`, standard Python operators.
    """
    # 1. Get Program IDs and define offsets
    # This corresponds to `blockIdx.x` and `blockIdx.y`
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    # Create blocks of offsets for Q, K, V
    # `offs_m` corresponds to the rows of the Q tile (Br)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # `offs_n` corresponds to the rows of the K/V tile (Bc)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # `offs_k` corresponds to the head dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 2. Initialize pointers to Q, K, V
    # This corresponds to calculating `Q_gmem_offset`, `K_gmem_offset`, etc.
    # The `pid_zh` is used to select the correct batch and head
    q_ptrs = Q + pid_zh * stride_qh + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :]
    k_ptrs = K + pid_zh * stride_kh + offs_n[None, :] * stride_kn + tl.arange(0, HEAD_DIM)[:, None]
    v_ptrs = V + pid_zh * stride_vh + offs_n[:, None] * stride_vn + tl.arange(0, HEAD_DIM)[None, :]
    
    # 3. Initialize accumulators for the online softmax
    # Corresponds to `R_D`, `lane_block_row_max_old`, `lane_block_row_sum_old`
    acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32)
    
    # Scale factor
    scale = (HEAD_DIM ** -0.5)

    # 4. Main loop over the sequence length of K and V
    # This is the `serial_range_for(tile_K_seqlen, 0, Tc, 1)` loop in CUDA
    for start_n in range(0, N_CTX, BLOCK_SIZE_N):
        # -- Load K and V tiles --
        # `tl.load` with masking handles boundary conditions automatically.
        # This replaces the complex `warp_copy_async` G2S -> S2R pipeline.
        current_k_ptrs = k_ptrs + start_n * stride_kn
        current_v_ptrs = v_ptrs + start_n * stride_vn
        
        k = tl.load(current_k_ptrs, mask=(start_n + offs_n)[None, :] < N_CTX, other=0.0)
        v = tl.load(current_v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # -- Compute S = Q @ K.T --
        # This part is significantly simplified. The CUDA code has an inner loop
        # over `kHeadDim` loading chunks into registers and calling `mma.sync`.
        # Triton's `tl.dot` abstracts this entire process.
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        s_ij = tl.dot(q, tl.trans(k)) * scale
        
        # -- Online Softmax Calculation --
        # This section replaces the manual max/sum reduction using warp shuffles
        # and the complex update logic.
        
        # a. Find new max
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # b. Calculate weights for updates
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # c. Calculate P_ij and update row-sum `l`
        p_ij = beta[:, None] * tl.exp(s_ij - m_new[:, None])
        l_ij = tl.sum(p_ij, axis=1)
        l_new = alpha * l_i + l_ij
        
        # -- Update Output Accumulator `acc` --
        # This corresponds to the P@V GEMM and the rescaling logic.
        
        # a. Rescale old accumulator
        acc = acc * alpha[:, None]
        
        # b. Compute O_ij = P_ij @ V and add to accumulator
        # `p_ij` must be cast to the same dtype as V for `tl.dot`
        p_ij = p_ij.to(V.dtype.element_ty)
        o_ij = tl.dot(p_ij, v)
        acc += o_ij
        
        # c. Update state for next iteration
        l_i = l_new
        m_i = m_new

    # 5. Final Normalization and Store
    # Corresponds to the final `rcp` and R2G store operation.
    # The complex `warp_shuffle_spread` for storing is handled by `tl.store`.
    acc = acc / l_i[:, None]
    
    # Initialize output pointers
    o_ptrs = O + pid_zh * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :]
    
    # Create a mask to avoid writing out of bounds
    out_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=out_mask)


def flash_attention(q, k, v):
    """
    Launcher function for the Flash Attention Triton kernel.
    
    This function sets up the grid, defines constants, and calls the kernel.
    It replaces the C++ `launch_flash_attn_...` and `PYBIND11_MODULE` parts.
    """
    # Input validation
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Head dimensions must match"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16, "Inputs must be FP16"

    # Tensor dimensions
    Z, H, N_CTX, HEAD_DIM = q.shape
    
    # Output tensor
    o = torch.empty_like(q)

    # Kernel configuration
    # These values are inspired by the CUDA template parameters and common Triton practices.
    # `Br` = 64 -> `BLOCK_SIZE_M` = 64
    # `Bc` = 32 -> `BLOCK_SIZE_N` = 32
    # `kHeadDim` is a multiple of 16 -> `BLOCK_SIZE_K` can be 32, 64, etc.
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64 # This is the inner dimension for the dot product loop
    
    # Heuristics for number of warps and stages
    num_warps = 4
    if HEAD_DIM >= 64:
        num_warps = 8
    
    # Define the execution grid
    # `ceil_div(QKV_seqlen, Br)` -> `triton.cdiv(N_CTX, BLOCK_SIZE_M)`
    # `QKV_batch * QKV_head` -> `Z * H`
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M), Z * H)

    # Launch the kernel
    _attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K, # Note: BLOCK_SIZE_K is not used in this simplified version
                                  # but is kept for compatibility with more complex patterns.
                                  # The tl.dot(q, tl.trans(k)) handles the full HEAD_DIM.
        num_warps=num_warps,
    )
    
    return o



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
    tri_out = flash_attention(q, k, v)
    print("max diff:", (ref_out - tri_out).abs().max().item())
    print("mean diff:", (ref_out - tri_out).abs().mean().item())
    print("ref value range:", ref_out.min().item(), ref_out.max().item())
    print("tri value range:", tri_out.min().item(), tri_out.max().item())
    
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
        fn = lambda: flash_attention(q, k, v)
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

'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_1.py
开始测试...
max diff: 0.03704833984375
mean diff: 0.00473785400390625
ref value range: -0.0672607421875 0.0716552734375
tri value range: -0.0733642578125 0.07330322265625
Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_1.py", line 265, in <module>
    test_non_causal_attention(4, 32, 1024, 64)
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/triton_after_annot_turn_1.py", line 213, in test_non_causal_attention
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
'''