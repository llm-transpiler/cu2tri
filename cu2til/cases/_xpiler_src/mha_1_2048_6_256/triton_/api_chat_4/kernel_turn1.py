import torch
import triton
import triton.language as tl
import math

@triton.jit
def mha_triton_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Stride information for each tensor
    stride_qz, stride_qq, stride_qh, stride_qd,
    stride_kz, stride_kq, stride_kk, stride_kd,
    stride_vz, stride_vq, stride_vk, stride_vd,
    stride_oz, stride_oq, stride_oh, stride_od,
    # Other metadata
    BATCH_SIZE: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN_K: tl.constexpr, # This was hardcoded as 6
    D_HEAD: tl.constexpr,    # This was hardcoded as 256
):
    """
    Triton kernel for a specific type of Multi-Head Attention.

    Each program instance computes the attention output for a single query vector.
    Grid: (SEQ_LEN_Q, BATCH_SIZE, NUM_HEADS)
    - pid_q: Index for the query sequence length dimension.
    - pid_z: Index for the batch dimension.
    - pid_h: Index for the head dimension.
    """
    # 1. Get the program IDs for the current instance
    pid_q = tl.program_id(0)  # Current query token index (j in CUDA code)
    pid_z = tl.program_id(1)  # Current batch index (i in CUDA code)
    pid_h = tl.program_id(2)  # Current head index (m in CUDA code)

    # 2. Calculate pointers to the start of the relevant data for this instance
    #    These pointers point to the beginning of the matrix/vector for the current (z, q, h)
    q_offset = pid_z * stride_qz + pid_q * stride_qq + pid_h * stride_qh
    # For K and V, the head dimension is replaced by the key/value sequence length
    k_offset = pid_z * stride_kz + pid_q * stride_kq
    v_offset = pid_z * stride_vz + pid_q * stride_vq
    out_offset = pid_z * stride_oz + pid_q * stride_oq + pid_h * stride_oh

    # 3. Load the query vector for the current head into SRAM
    #    q is a vector of size [D_HEAD]
    offs_d = tl.arange(0, D_HEAD)
    q_ptrs = Q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs)

    # 4. Compute the score vector S = Q @ K.T
    #    - Load the entire K matrix for this query position: [SEQ_LEN_K, D_HEAD]
    #    - Compute dot product with the Q vector
    offs_k = tl.arange(0, SEQ_LEN_K)
    k_ptrs = K_ptr + k_offset + (offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd)
    k_mat = tl.load(k_ptrs)
    
    # scores will be a vector of size [SEQ_LEN_K]
    scores = tl.dot(k_mat, q)

    # 5. Apply scaling and softmax
    scaling_factor = 1.0 / math.sqrt(D_HEAD)
    scores = scores * scaling_factor
    p = tl.softmax(scores) # `p` is the attention probability vector, size [SEQ_LEN_K]

    # 6. Compute the final output vector O = P @ V
    #    - Load the entire V matrix for this query position: [SEQ_LEN_K, D_HEAD]
    #    - Compute dot product with the attention probability vector `p`
    v_ptrs = V_ptr + v_offset + (offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd)
    v_mat = tl.load(v_ptrs)
    
    # Ensure `p` has the same dtype as `v_mat` for the dot product
    p = p.to(v_mat.dtype)
    
    # `output_vec` will be a vector of size [D_HEAD]
    # tl.dot((1, SEQ_LEN_K), (SEQ_LEN_K, D_HEAD)) -> (1, D_HEAD)
    output_vec = tl.dot(p[None, :], v_mat)

    # 7. Store the resulting output vector to global memory
    out_ptrs = Out_ptr + out_offset + offs_d * stride_od
    tl.store(out_ptrs, output_vec)


def mha_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the Triton MHA kernel.

    Args:
        q (torch.Tensor): Query tensor of shape (BATCH, SEQ_LEN_Q, NUM_HEADS, D_HEAD)
        k (torch.Tensor): Key tensor of shape (BATCH, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD)
        v (torch.Tensor): Value tensor of shape (BATCH, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD)

    Returns:
        torch.Tensor: Output tensor of shape (BATCH, SEQ_LEN_Q, NUM_HEADS, D_HEAD)
    """
    # Ensure tensors are on the GPU and have the correct layout
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    # Extract dimensions from input tensors
    BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, D_HEAD = q.shape
    _, _, SEQ_LEN_K, _ = k.shape

    # Create the output tensor
    output = torch.empty_like(q)

    # Define the grid for launching the kernel
    # Each program instance handles one query position for one head
    grid = (SEQ_LEN_Q, BATCH_SIZE, NUM_HEADS)

    # Launch the Triton kernel
    mha_triton_kernel[grid](
        # Pointers
        q, k, v, output,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for Output
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Metadata (as tl.constexpr)
        BATCH_SIZE=BATCH_SIZE,
        SEQ_LEN_Q=SEQ_LEN_Q,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN_K=SEQ_LEN_K,
        D_HEAD=D_HEAD,
    )
    return output

# Example Usage:
if __name__ == '__main__':
    # Match the dimensions from the CUDA code
    BATCH_SIZE = 4
    SEQ_LEN_Q = 2048
    NUM_HEADS = 6
    SEQ_LEN_K = 6   # The hardcoded '6' in the loops
    D_HEAD = 256

    # Create random input tensors on the GPU
    q = torch.randn(BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, D_HEAD, device='cuda', dtype=torch.float32)
    # Note the shape of K and V, matching the unusual indexing from the CUDA code
    k = torch.randn(BATCH_SIZE, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD, device='cuda', dtype=torch.float32)
    v = torch.randn(BATCH_SIZE, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD, device='cuda', dtype=torch.float32)

    # Run the Triton kernel
    triton_output = mha_wrapper(q, k, v)

    # --- For verification, here is a PyTorch implementation of the same logic ---
    def mha_pytorch_reference(q, k, v):
        scaling_factor = 1.0 / math.sqrt(q.shape[-1])
        # Permute to put head dimension first for easier broadcasting
        # Q: [B, H, Q_LEN, D]
        # K: [B, Q_LEN, K_LEN, D] -> [B, 1, Q_LEN, K_LEN, D]
        # V: [B, Q_LEN, K_LEN, D] -> [B, 1, Q_LEN, K_LEN, D]
        q_ref = q.permute(0, 2, 1, 3)
        k_ref = k.unsqueeze(1)
        v_ref = v.unsqueeze(1)

        # S = Q @ K.T
        # q_ref[:, :, :, None, :] -> [B, H, Q_LEN, 1, D]
        # k_ref.transpose(-2, -1) -> [B, 1, Q_LEN, D, K_LEN]
        # scores -> [B, H, Q_LEN, 1, K_LEN]
        scores = torch.matmul(q_ref[:, :, :, None, :], k_ref.transpose(-2, -1)) * scaling_factor
        
        p = torch.softmax(scores, dim=-1) # Softmax over K_LEN dimension

        # O = P @ V
        # p -> [B, H, Q_LEN, 1, K_LEN]
        # v_ref -> [B, 1, Q_LEN, K_LEN, D]
        # output -> [B, H, Q_LEN, 1, D]
        output = torch.matmul(p, v_ref)
        
        # Squeeze and permute back to original shape: [B, Q_LEN, H, D]
        return output.squeeze(-2).permute(0, 2, 1, 3)

    pytorch_output = mha_pytorch_reference(q, k, v)

    # Compare results
    print("Triton and PyTorch outputs are close:", torch.allclose(triton_output, pytorch_output, atol=1e-4, rtol=1e-4))
    # print(triton_output)
    # print(pytorch_output)
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 1654, in arange
    return _semantic.arange(start, end)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 583, in arange
    raise ValueError("arange's range must be a power of 2")
ValueError: arange's range must be a power of 2

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 147, in <module>
    triton_output = mha_wrapper(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 111, in mha_wrapper
    mha_triton_kernel[grid](
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
triton.compiler.errors.CompilationError: at 47:13:
    out_offset = pid_z * stride_oz + pid_q * stride_oq + pid_h * stride_oh

    # 3. Load the query vector for the current head into SRAM
    #    q is a vector of size [D_HEAD]
    offs_d = tl.arange(0, D_HEAD)
    q_ptrs = Q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs)

    # 4. Compute the score vector S = Q @ K.T
    #    - Load the entire K matrix for this query position: [SEQ_LEN_K, D_HEAD]
    #    - Compute dot product with the Q vector
    offs_k = tl.arange(0, SEQ_LEN_K)
             ^
'''