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
        # D_HEAD is a constexpr (Python int), not a tl.tensor. Use it directly.
        s_ij *= (1.0 / tl.sqrt(D_HEAD))
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

# Example Usage
if __name__ == '__main__':
    Z, H, N_CTX, D_HEAD = 4, 12, 2048, 64

    # Create sample tensors
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device='cuda')

    # Run Triton implementation
    triton_output = flash_attn_triton(q, k, v)

    # For verification, run a PyTorch equivalent (requires a lot of memory)
    try:
        pytorch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        print("Comparing Triton output to PyTorch SDPA...")
        # Note: Due to different computation orders and precision, small differences are expected.
        # The tolerance `atol` might need adjustment based on the problem size.
        assert torch.allclose(triton_output, pytorch_output, atol=1e-1, rtol=1e-2)
        print("✅ Triton and PyTorch outputs match!")
    except torch.cuda.OutOfMemoryError:
        print("Could not run PyTorch reference due to OOM. Verification skipped.")
    except Exception as e:
        print(f"An error occurred during verification: {e}")

    print("\nTriton Output (first 4x4 of first head/batch):")
    print(triton_output[0, 0, :4, :4])
'''
root@ubuntu-ThinkStation-P520:/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some# python reducecode_direct_2_turn2.py 
/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn2.py:96:31: error: 'math.sqrt' op operand #0 must be floating-point-like, but got 'i32'
        s_ij *= (1.0 / tl.sqrt(D_HEAD))
                              ^
"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {tt.divisibility = 16 : i32}], function_type = (!tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "_flash_attn_forward_kernel", sym_visibility = "public"}> ({
  ^bb0(%arg10: !tt.ptr<f16>, %arg11: !tt.ptr<f16>, %arg12: !tt.ptr<f16>, %arg13: !tt.ptr<f16>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32):
    %16 = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32
    %17 = "tt.get_program_id"() <{axis = 1 : i32}> : () -> i32
    %18 = "arith.divsi"(%17, %arg27) : (i32, i32) -> i32
    %19 = "arith.remsi"(%17, %arg27) : (i32, i32) -> i32
    %20 = "arith.extsi"(%18) : (i32) -> i64
    %21 = "arith.extsi"(%arg14) : (i32) -> i64
    %22 = "arith.muli"(%20, %21) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %23 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %24 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %25 = "arith.cmpi"(%22, %23) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %26 = "arith.cmpi"(%22, %24) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %27 = "arith.andi"(%25, %26) : (i1, i1) -> i1
    %28 = "arith.muli"(%18, %arg14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %29 = "tt.addptr"(%arg10, %28) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %30 = "arith.extsi"(%19) : (i32) -> i64
    %31 = "arith.extsi"(%arg15) : (i32) -> i64
    %32 = "arith.muli"(%30, %31) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %33 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %34 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %35 = "arith.cmpi"(%32, %33) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %36 = "arith.cmpi"(%32, %34) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %37 = "arith.andi"(%35, %36) : (i1, i1) -> i1
    %38 = "arith.muli"(%19, %arg15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %39 = "tt.addptr"(%29, %38) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %40 = "arith.extsi"(%18) : (i32) -> i64
    %41 = "arith.extsi"(%arg17) : (i32) -> i64
    %42 = "arith.muli"(%40, %41) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %43 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %44 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %45 = "arith.cmpi"(%42, %43) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %46 = "arith.cmpi"(%42, %44) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %47 = "arith.andi"(%45, %46) : (i1, i1) -> i1
    %48 = "arith.muli"(%18, %arg17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %49 = "tt.addptr"(%arg11, %48) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %50 = "arith.extsi"(%19) : (i32) -> i64
    %51 = "arith.extsi"(%arg18) : (i32) -> i64
    %52 = "arith.muli"(%50, %51) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %53 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %54 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %55 = "arith.cmpi"(%52, %53) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %56 = "arith.cmpi"(%52, %54) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %57 = "arith.andi"(%55, %56) : (i1, i1) -> i1
    %58 = "arith.muli"(%19, %arg18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %59 = "tt.addptr"(%49, %58) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %60 = "arith.extsi"(%18) : (i32) -> i64
    %61 = "arith.extsi"(%arg20) : (i32) -> i64
    %62 = "arith.muli"(%60, %61) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %63 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %64 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %65 = "arith.cmpi"(%62, %63) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %66 = "arith.cmpi"(%62, %64) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %67 = "arith.andi"(%65, %66) : (i1, i1) -> i1
    %68 = "arith.muli"(%18, %arg20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %69 = "tt.addptr"(%arg12, %68) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %70 = "arith.extsi"(%19) : (i32) -> i64
    %71 = "arith.extsi"(%arg21) : (i32) -> i64
    %72 = "arith.muli"(%70, %71) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %73 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %74 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %75 = "arith.cmpi"(%72, %73) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %76 = "arith.cmpi"(%72, %74) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %77 = "arith.andi"(%75, %76) : (i1, i1) -> i1
    %78 = "arith.muli"(%19, %arg21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %79 = "tt.addptr"(%69, %78) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %80 = "arith.extsi"(%18) : (i32) -> i64
    %81 = "arith.extsi"(%arg23) : (i32) -> i64
    %82 = "arith.muli"(%80, %81) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %83 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %84 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %85 = "arith.cmpi"(%82, %83) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %86 = "arith.cmpi"(%82, %84) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %87 = "arith.andi"(%85, %86) : (i1, i1) -> i1
    %88 = "arith.muli"(%18, %arg23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %89 = "tt.addptr"(%arg13, %88) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %90 = "arith.extsi"(%19) : (i32) -> i64
    %91 = "arith.extsi"(%arg24) : (i32) -> i64
    %92 = "arith.muli"(%90, %91) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %93 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %94 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %95 = "arith.cmpi"(%92, %93) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %96 = "arith.cmpi"(%92, %94) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %97 = "arith.andi"(%95, %96) : (i1, i1) -> i1
    %98 = "arith.muli"(%19, %arg24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %99 = "tt.addptr"(%89, %98) : (!tt.ptr<f16>, i32) -> !tt.ptr<f16>
    %100 = "tt.call"() <{callee = @"triton.language.standard.zeros____(0, 0)cconstexpr_64__(0, 1)cconstexpr_64__(1,)cconstexpr_fp32_"}> : () -> tensor<64x64xf32>
    %101 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32
    %102 = "arith.constant"() <{value = dense<0xFF800000> : tensor<64xf32>}> : () -> tensor<64xf32>
    %103 = "tt.call"() <{callee = @"triton.language.standard.zeros____(0, 0)cconstexpr_64__(1,)cconstexpr_fp32_"}> : () -> tensor<64xf32>
    %104 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %105 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %106 = "arith.extsi"(%16) : (i32) -> i64
    %107 = "arith.extsi"(%105) : (i32) -> i64
    %108 = "arith.muli"(%106, %107) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %109 = "arith.constant"() <{value = 2147483647 : i64}> : () -> i64
    %110 = "arith.constant"() <{value = -2147483648 : i64}> : () -> i64
    %111 = "arith.cmpi"(%108, %109) <{predicate = 3 : i64}> : (i64, i64) -> i1
    %112 = "arith.cmpi"(%108, %110) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %113 = "arith.andi"(%111, %112) : (i1, i1) -> i1
    %114 = "arith.muli"(%16, %105) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %115 = "arith.extsi"(%arg28) : (i32) -> i64
    %116 = "arith.constant"() <{value = 64 : i64}> : () -> i64
    %117 = "arith.extsi"(%arg16) : (i32) -> i64
    %118 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %119 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %120 = "tt.make_tensor_ptr"(%39, %115, %116, %117, %118, %114, %119) <{order = array<i32: 1, 0>}> : (!tt.ptr<f16>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<64x64xf16>>
    %121 = "tt.load"(%120) <{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<tensor<64x64xf16>>) -> tensor<64x64xf16>
    %122 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %123 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %124 = "arith.bitcast"(%122) : (i32) -> i32
    %125 = "arith.bitcast"(%arg28) : (i32) -> i32
    %126 = "arith.bitcast"(%123) : (i32) -> i32
    %127 = "ub.poison"() <{value = #ub.poison}> : () -> i32
    %128:3 = "scf.for"(%124, %125, %126, %100, %103, %102) ({
    ^bb0(%arg29: i32, %arg30: tensor<64x64xf32>, %arg31: tensor<64xf32>, %arg32: tensor<64xf32>):
      %139 = "arith.constant"() <{value = 64 : i64}> : () -> i64
      %140 = "arith.extsi"(%arg28) : (i32) -> i64
      %141 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %142 = "arith.extsi"(%arg19) : (i32) -> i64
      %143 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %144 = "tt.make_tensor_ptr"(%59, %139, %140, %141, %142, %143, %arg29) <{order = array<i32: 0, 1>}> : (!tt.ptr<f16>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<64x32xf16>>
      %145 = "arith.extsi"(%arg28) : (i32) -> i64
      %146 = "arith.constant"() <{value = 64 : i64}> : () -> i64
      %147 = "arith.extsi"(%arg22) : (i32) -> i64
      %148 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %149 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %150 = "tt.make_tensor_ptr"(%79, %145, %146, %147, %148, %arg29, %149) <{order = array<i32: 1, 0>}> : (!tt.ptr<f16>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<32x64xf16>>
      %151 = "tt.load"(%144) <{boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<tensor<64x32xf16>>) -> tensor<64x32xf16>
      %152 = "tt.load"(%150) <{boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<tensor<32x64xf16>>) -> tensor<32x64xf16>
      %153 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %154 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x32xf32>}> : () -> tensor<64x32xf32>
      %155 = "tt.dot"(%121, %151, %154) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x64xf16>, tensor<64x32xf16>, tensor<64x32xf32>) -> tensor<64x32xf32>
      %156 = "arith.constant"() <{value = 64 : i32}> : () -> i32
      %157 = "math.sqrt"(%156) <{fastmath = #arith.fastmath<none>}> : (i32) -> i32
      %158 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %159 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %160 = "arith.sitofp"(%157) : (i32) -> f32
      %161 = "arith.divf"(%159, %160) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %162 = "tt.splat"(%161) : (f32) -> tensor<64x32xf32>
      %163 = "arith.mulf"(%155, %162) <{fastmath = #arith.fastmath<none>}> : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
      %164 = "tt.call"(%163) <{callee = @"triton.language.standard.max__fp32S64_32S__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cconstexpr_True__(4,)cconstexpr_False_"}> : (tensor<64x32xf32>) -> tensor<64xf32>
      %165 = "arith.maxnumf"(%arg32, %164) <{fastmath = #arith.fastmath<none>}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %166 = "arith.subf"(%arg32, %165) <{fastmath = #arith.fastmath<none>}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %167 = "math.exp"(%166) <{fastmath = #arith.fastmath<none>}> : (tensor<64xf32>) -> tensor<64xf32>
      %168 = "tt.expand_dims"(%165) <{axis = 1 : i32}> : (tensor<64xf32>) -> tensor<64x1xf32>
      %169 = "tt.broadcast"(%168) : (tensor<64x1xf32>) -> tensor<64x32xf32>
      %170 = "arith.subf"(%163, %169) <{fastmath = #arith.fastmath<none>}> : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
      %171 = "math.exp"(%170) <{fastmath = #arith.fastmath<none>}> : (tensor<64x32xf32>) -> tensor<64x32xf32>
      %172 = "arith.mulf"(%167, %arg31) <{fastmath = #arith.fastmath<none>}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %173 = "tt.call"(%171) <{callee = @"triton.language.standard.sum__fp32S64_32S__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cNone"}> : (tensor<64x32xf32>) -> tensor<64xf32>
      %174 = "arith.addf"(%172, %173) <{fastmath = #arith.fastmath<none>}> : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %175 = "tt.expand_dims"(%167) <{axis = 1 : i32}> : (tensor<64xf32>) -> tensor<64x1xf32>
      %176 = "tt.broadcast"(%175) : (tensor<64x1xf32>) -> tensor<64x64xf32>
      %177 = "arith.mulf"(%arg30, %176) <{fastmath = #arith.fastmath<none>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %178 = "arith.truncf"(%171) : (tensor<64x32xf32>) -> tensor<64x32xf16>
      %179 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %180 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x64xf32>}> : () -> tensor<64x64xf32>
      %181 = "tt.dot"(%178, %152, %180) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %182 = "arith.addf"(%177, %181) <{fastmath = #arith.fastmath<none>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      "scf.yield"(%182, %174, %165) : (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> ()
    }) : (i32, i32, i32, tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)
    %129 = "tt.expand_dims"(%128#1) <{axis = 1 : i32}> : (tensor<64xf32>) -> tensor<64x1xf32>
    %130 = "tt.broadcast"(%129) : (tensor<64x1xf32>) -> tensor<64x64xf32>
    %131 = "arith.divf"(%128#0, %130) <{fastmath = #arith.fastmath<none>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %132 = "arith.extsi"(%arg28) : (i32) -> i64
    %133 = "arith.constant"() <{value = 64 : i64}> : () -> i64
    %134 = "arith.extsi"(%arg25) : (i32) -> i64
    %135 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %136 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %137 = "tt.make_tensor_ptr"(%99, %132, %133, %134, %135, %114, %136) <{order = array<i32: 1, 0>}> : (!tt.ptr<f16>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<64x64xf16>>
    %138 = "arith.truncf"(%131) : (tensor<64x64xf32>) -> tensor<64x64xf16>
    "tt.store"(%137, %138) <{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32}> : (!tt.ptr<tensor<64x64xf16>>, tensor<64x64xf16>) -> ()
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = () -> tensor<64x64xf32>, sym_name = "triton.language.standard.zeros____(0, 0)cconstexpr_64__(0, 1)cconstexpr_64__(1,)cconstexpr_fp32_", sym_visibility = "private"}> ({
    %13 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %14 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64x64xf32>}> : () -> tensor<64x64xf32>
    "tt.return"(%14) : (tensor<64x64xf32>) -> ()
  ^bb1:  // no predecessors
    %15 = "ub.poison"() <{value = #ub.poison}> : () -> tensor<64x64xf32>
    "tt.return"(%15) : (tensor<64x64xf32>) -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = () -> tensor<64xf32>, sym_name = "triton.language.standard.zeros____(0, 0)cconstexpr_64__(1,)cconstexpr_fp32_", sym_visibility = "private"}> ({
    %10 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %11 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    "tt.return"(%11) : (tensor<64xf32>) -> ()
  ^bb1:  // no predecessors
    %12 = "ub.poison"() <{value = #ub.poison}> : () -> tensor<64xf32>
    "tt.return"(%12) : (tensor<64xf32>) -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = (tensor<64x32xf32>) -> tensor<64xf32>, sym_name = "triton.language.standard.max__fp32S64_32S__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cconstexpr_True__(4,)cconstexpr_False_", sym_visibility = "private"}> ({
  ^bb0(%arg7: tensor<64x32xf32>):
    %7 = "tt.reduce"(%arg7) <{axis = 1 : i32}> ({
    ^bb0(%arg8: f32, %arg9: f32):
      %9 = "tt.call"(%arg8, %arg9) <{callee = @triton.language.standard._elementwise_max__fp32_fp32__}> : (f32, f32) -> f32
      "tt.reduce.return"(%9) : (f32) -> ()
    }) : (tensor<64x32xf32>) -> tensor<64xf32>
    "tt.return"(%7) : (tensor<64xf32>) -> ()
  ^bb1:  // no predecessors
    %8 = "ub.poison"() <{value = #ub.poison}> : () -> tensor<64xf32>
    "tt.return"(%8) : (tensor<64xf32>) -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = (f32, f32) -> f32, sym_name = "triton.language.standard._elementwise_max__fp32_fp32__", sym_visibility = "private"}> ({
  ^bb0(%arg5: f32, %arg6: f32):
    %5 = "arith.maxnumf"(%arg5, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "tt.return"(%5) : (f32) -> ()
  ^bb1:  // no predecessors
    %6 = "ub.poison"() <{value = #ub.poison}> : () -> f32
    "tt.return"(%6) : (f32) -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = (tensor<64x32xf32>) -> tensor<64xf32>, sym_name = "triton.language.standard.sum__fp32S64_32S__(1,)cconstexpr_1__(2,)cconstexpr_False__(3,)cNone", sym_visibility = "private"}> ({
  ^bb0(%arg2: tensor<64x32xf32>):
    %2 = "tt.reduce"(%arg2) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = "tt.call"(%arg3, %arg4) <{callee = @triton.language.standard._sum_combine__fp32_fp32__}> : (f32, f32) -> f32
      "tt.reduce.return"(%4) : (f32) -> ()
    }) : (tensor<64x32xf32>) -> tensor<64xf32>
    "tt.return"(%2) : (tensor<64xf32>) -> ()
  ^bb1:  // no predecessors
    %3 = "ub.poison"() <{value = #ub.poison}> : () -> tensor<64xf32>
    "tt.return"(%3) : (tensor<64xf32>) -> ()
  }) {noinline = false} : () -> ()
  "tt.func"() <{function_type = (f32, f32) -> f32, sym_name = "triton.language.standard._sum_combine__fp32_fp32__", sym_visibility = "private"}> ({
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = "arith.addf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "tt.return"(%0) : (f32) -> ()
  ^bb1:  // no predecessors
    %1 = "ub.poison"() <{value = #ub.poison}> : () -> f32
    "tt.return"(%1) : (f32) -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 }, triton-rewrite-tensor-pointer, triton-rewrite-tensor-descriptor-to-pointer, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-combine, triton-reorder-broadcast, cse, symbol-dce, triton-loop-unroll)",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn2.py:6:0: error: Failures have been detected while processing an MLIR pass pipeline
/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn2.py:6:0: note: Pipeline failed while executing [`Inliner` on 'builtin.module' operation, `Canonicalizer` on 'tt.func' operation: @_flash_attn_forward_kernel]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
Traceback (most recent call last):
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn2.py", line 191, in <module>
    triton_output = flash_attn_triton(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/fa/cuda/flash_attn_mma_stages_split_q_shared_kv/some/reducecode_direct_2_turn2.py", line 166, in flash_attn_triton
    _flash_attn_forward_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 390, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 359, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/compiler.py", line 455, in <lambda>
    stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/compiler.py", line 226, in make_ttir
    pm.run(mod)
RuntimeError: PassManager::run failed
'''