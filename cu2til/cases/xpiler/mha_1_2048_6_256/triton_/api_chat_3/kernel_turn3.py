import torch
import triton
import triton.language as tl

# This is a direct translation of the INTENT of the original CUDA code,
# implementing a standard and correct Multi-Head Attention mechanism.
# The original CUDA code had a logical bug in its attention score calculation.

@triton.jit
def mha_triton_kernel(
    # Pointers to matrices
    Q, K, V, O,
    # Stride variables for tensors
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Other parameters
    BATCH, NUM_HEADS, SEQ_LEN,
    D_HEAD: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Triton Kernel for Fused Multi-Head Attention.
    Computes O = softmax(Q @ K.T / sqrt(D_HEAD)) @ V
    """
    # 1. Get Program IDs to determine which part of the work this program instance will do
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS

    # 2. Create pointers to the first element of Q, K, V for this instance
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh

    # 3. Initialize offsets for the Q, K, V tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # 4. Load the Q tile (BLOCK_M x D_HEAD)
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :])
    mask_m = offs_m < SEQ_LEN
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 5. Initialize accumulator and online softmax statistics
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    sm_scale = 1.0 / (D_HEAD ** 0.5)

    # 6. Main loop over the key/value sequence in blocks of BLOCK_N
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # --- Load K and V for the current block ---
        current_offs_n = start_n + offs_n
        
        k_ptrs = k_ptr + (current_offs_n[None, :] * stride_kn + offs_d[:, None])
        mask_n = current_offs_n < SEQ_LEN
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        v_ptrs = v_ptr + (current_offs_n[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # --- Compute attention scores (S = Q @ K.T) ---
        s = tl.dot(q, k) * sm_scale
        
        # --- Online Softmax Calculation ---
        m_curr = tl.max(s, 1)
        m_new = tl.maximum(m_i, m_curr)
        
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = tl.exp(s - m_new[:, None])
        l_i += tl.sum(p, 1)

        # --- Update accumulator with weighted V ---
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # 7. Finalize the output
    acc = acc / l_i[:, None]

    # 8. Write the final output tile to global memory
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def mha(q, k, v):
    """
    Python wrapper for the Triton MHA kernel.
    Assumes input tensors are in shape (BATCH, NUM_HEADS, SEQ_LEN, D_HEAD).
    """
    BATCH, NUM_HEADS, SEQ_LEN, D_HEAD = q.shape
    
    o = torch.empty_like(q)

    # --- FIX IS HERE and VERIFIED ---
    # Reduced block sizes to fit within shared memory limits.
    # BLOCK_M = 64
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), BATCH * NUM_HEADS)

    mha_triton_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BATCH, NUM_HEADS, SEQ_LEN, D_HEAD,
        # Meta-parameters (constants for the compiler)
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D_HEAD,
        BLOCK_N=BLOCK_N,
    )
    return o

def test_mha():
    # Problem dimensions from the original CUDA code
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    # Create random input tensors
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda', dtype=torch.float16)

    # --- PyTorch reference implementation ---
    def torch_mha(q, k, v):
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    # --- Run both implementations ---
    print("Running Triton MHA...")
    triton_output = mha(q, k, v)

    print("Running PyTorch reference MHA...")
    torch_output = torch_mha(q, k, v)

    # --- Compare results ---
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and PyTorch outputs match!")
    else:
        print("❌ Triton and PyTorch outputs DO NOT match!")
        max_diff = (triton_output - torch_output).abs().max().item()
        print(f"   Max absolute difference: {max_diff}")

if __name__ == "__main__":
    test_mha()
'''
在hopper上
root@sigma106:/workspace# /usr/bin/python /workspace/cases/mha/test.py
Running Triton MHA...
Running PyTorch reference MHA...
Triton output shape: torch.Size([4, 6, 2048, 256])
PyTorch output shape: torch.Size([4, 6, 2048, 256])
✅ Triton and PyTorch outputs match!
'''