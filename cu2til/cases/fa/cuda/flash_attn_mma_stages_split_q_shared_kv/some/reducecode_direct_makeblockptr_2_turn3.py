import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_forward_kernel(
    # Input/Output Pointers
    Q, K, V, O,
    # Stride information for Tensors
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other metadata
    Z, H, N_CTX,
    # Kernel-specific constants
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for Flash Attention.
    This is a translation of the provided CUDA kernel, using tl.make_block_ptr.
    """
    # 1. Get Program and Block IDs
    # This thread block processes the `start_m`-th block of Q rows
    start_m = tl.program_id(0)
    # This thread block processes the `off_hz`-th head/batch combination
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # 2. Initialize pointers to Q, K, V
    # These are the base pointers for the current head and batch
    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh
    k_base_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_base_ptr = V + off_z * stride_vz + off_h * stride_vh
    o_base_ptr = O + off_z * stride_oz + off_h * stride_oh

    # 3. Initialize accumulators for online softmax
    # Corresponds to `lane_block_row_max_old`, `lane_block_row_sum_old`, and `R_D` in CUDA
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 4. Load the Q tile from Global Memory
    # This corresponds to the initial `cp.async` load of the Q tile into smem
    q_offset_m = start_m * BLOCK_M
    
    # Create block pointers for Q
    q_ptrs = tl.make_block_ptr(
        base=q_base_ptr,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(q_offset_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    # Load Q. The mask prevents out-of-bounds access for the last block.
    q = tl.load(q_ptrs, boundary_check=(0, 1))

    # 5. Main loop over K and V tiles
    # Corresponds to `for (int tile_K_seqlen = 0; ...)` in CUDA
    # The original CUDA code is not causal, so we iterate over the full sequence.
    for start_n in range(0, N_CTX, BLOCK_N):
        # --- Load K and V for the current block ---
        # Create block pointers for K and V
        k_ptrs = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(D_HEAD, N_CTX), # Note: shape is (D_HEAD, N_CTX) for transposed matmul
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(D_HEAD, BLOCK_N),
            order=(0, 1)
        )
        v_ptrs = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(N_CTX, D_HEAD),
            strides=(stride_vn, stride_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0)
        )
        # Load K and V tiles. Boundary checks handle sequences not divisible by BLOCK_N.
        k = tl.load(k_ptrs, boundary_check=(1,))
        v = tl.load(v_ptrs, boundary_check=(0,))

        # --- Compute S = Q @ K^T ---
        # Corresponds to the first `mma.sync` loop in CUDA
        s_ij = tl.dot(q, k)
        
        # ----------- FIX IS HERE -----------
        # The CUDA code applies scale during the softmax step. Applying it here is equivalent.
        # D_HEAD must be explicitly cast to float for tl.sqrt.
        s_ij *= (1.0 / tl.sqrt(float(D_HEAD)))
        # -----------------------------------

        # --- Online Softmax Calculation ---
        # This block replaces the manual max reduction, exp, and sum reduction in CUDA
        
        # a. Find new max
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # b. Calculate P_ij and rescale l_i, acc
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)
        
        # c. Update accumulator
        # Rescale previous accumulator value
        acc *= alpha[:, None]
        # Add the contribution of the current tile: P_ij @ V
        # Corresponds to the second `mma.sync` loop in CUDA
        acc += tl.dot(p_ij.to(Q.dtype.element_ty), v)
        
        # d. Update m_i and l_i for the next iteration
        l_i = l_new
        m_i = m_new

    # 6. Final Rescaling and Store to Output
    # Corresponds to the final rescaling and store logic in CUDA
    # Normalize the accumulator
    acc = acc / l_i[:, None]

    # Create block pointer for the output tensor O
    o_ptrs = tl.make_block_ptr(
        base=o_base_ptr,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_ok),
        offsets=(q_offset_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    # Store the final result.
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), boundary_check=(0, 1))


def flash_attn_triton(Q, K, V):
    """
    Python launcher for the Triton Flash Attention kernel.
    """
    # 1. Input validation
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape
    
    Z, H, N_CTX, D_HEAD = Q.shape
    
    # 2. Output tensor
    O = torch.empty_like(Q)

    # 3. Kernel configuration
    # These values are taken from the CUDA kernel's `Br` and `Bc` constants
    BLOCK_M = 64
    BLOCK_N = 32
    
    # The grid is 2D:
    # - dim 0: iterates over the Q sequence length
    # - dim 1: iterates over batches and heads
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    # 4. Launch kernel
    _flash_attn_forward_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD, # BLOCK_DMODEL is an alias for D_HEAD
    )
    
    return O

# # Example Usage
# if __name__ == '__main__':
#     Z, H, N_CTX, D_HEAD = 4, 12, 2048, 64

#     # Create sample tensors
#     q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
#     k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
#     v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')

#     # Run Triton implementation
#     triton_output = flash_attn_triton(q, k, v)

#     # For verification, run a PyTorch equivalent (requires a lot of memory)
#     try:
#         pytorch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
#         print("Comparing Triton output to PyTorch SDPA...")
#         # Note: Due to different computation orders and precision, small differences are expected.
#         # The tolerance `atol` might need adjustment based on the problem size.
#         assert torch.allclose(triton_output, pytorch_output, atol=1e-1, rtol=1e-2)
#         print("✅ Triton and PyTorch outputs match!")
#     except torch.cuda.OutOfMemoryError:
#         print("Could not run PyTorch reference due to OOM. Verification skipped.")
#     except Exception as e:
#         print(f"An error occurred during verification: {e}")

#     print("\nTriton Output (first 4x4 of first head/batch):")
#     print(triton_output[0, 0, :4, :4])
# '''
# root@ubuntu-ThinkStation-P520:/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some# python reducecode_direct_makeblockptr_2_turn3.py 
# Comparing Triton output to PyTorch SDPA...
# ✅ Triton and PyTorch outputs match!

# Triton Output (first 4x4 of first head/batch):
# tensor([[-0.0581,  0.0198,  0.0228,  0.0014],
#         [ 0.0626, -0.0622,  0.0790, -0.0444],
#         [ 0.0433,  0.0284,  0.0096,  0.0888],
#         [-0.0303,  0.0115,  0.0492, -0.0391]], device='cuda:0',
#        dtype=torch.float16)
# '''



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
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_makeblockptr_2_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     142.120018  141.346863
1   2048.0      99.983940  114.458813
2   4096.0     102.917370  120.394536
3   8192.0     101.041411  126.641404
4  16384.0     104.727207  129.894676
基准测试完成！
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_makeblockptr_2_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     148.499607  139.662966
1   2048.0     103.865721  114.377729
2   4096.0     102.935599  120.682817
3   8192.0     100.144966  127.542472
4  16384.0     105.937411  132.124388
基准测试完成！
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_makeblockptr_2_turn3.py
开始测试...
Non-casual attention pass.
测试完成！
开始基准测试...
non-causal-attention-batch4-head32-d64-fwd:
   SEQ_LEN  Triton [FP16]     Flash-2
0   1024.0     146.620801  150.167821
1   2048.0     106.774998  113.983026
2   4096.0     102.909768  120.365650
3   8192.0     105.157461  128.401156
4  16384.0     106.388752  131.390779
基准测试完成！
'''