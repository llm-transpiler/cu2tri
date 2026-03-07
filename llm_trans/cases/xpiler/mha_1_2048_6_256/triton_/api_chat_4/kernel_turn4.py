import torch
import triton
import triton.language as tl
import math

def next_power_of_2(n):
    if n == 0: return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

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
    SEQ_LEN_K: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_offset = pid_z * stride_qz + pid_q * stride_qq + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_q * stride_kq
    v_offset = pid_z * stride_vz + pid_q * stride_vq
    out_offset = pid_z * stride_oz + pid_q * stride_oq + pid_h * stride_oh

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    d_mask = offs_d < D_HEAD
    k_mask = offs_k < SEQ_LEN_K

    q_ptrs = Q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    k_ptrs = K_ptr + k_offset + (offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd)
    load_mask_2d = k_mask[:, None] & d_mask[None, :]
    k_mat = tl.load(k_ptrs, mask=load_mask_2d, other=0.0)
    
    # --- FIX: Replace tl.dot with manual implementation for small dimensions ---
    # tl.dot requires dimensions >= 16. Here, one dimension is 1 or 8.
    # We implement the dot product as an element-wise product followed by a reduction.
    # (K, D) * (1, D) -> (K, D), then sum over axis=1 -> (K,)
    scores = tl.sum(k_mat * q[None, :], axis=1)

    scaling_factor = 1.0 / math.sqrt(D_HEAD)
    scores = scores * scaling_factor
    
    # `scores` is now 1D, so we use the 1D `k_mask`
    scores = tl.where(k_mask, scores, -float('inf'))
    p = tl.softmax(scores) # `p` is also a 1D vector [BLOCK_SIZE_K]

    v_ptrs = V_ptr + v_offset + (offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd)
    v_mat = tl.load(v_ptrs, mask=load_mask_2d, other=0.0)
    
    p = p.to(v_mat.dtype)

    # --- FIX: Replace the second tl.dot as well ---
    # (K, 1) * (K, D) -> (K, D), then sum over axis=0 -> (D,)
    output_vec = tl.sum(p[:, None] * v_mat, axis=0)
    
    out_ptrs = Out_ptr + out_offset + offs_d * stride_od
    tl.store(out_ptrs, output_vec, mask=d_mask)


def mha_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, D_HEAD = q.shape
    _, _, SEQ_LEN_K, _ = k.shape

    output = torch.empty_like(q)
    grid = (SEQ_LEN_Q, BATCH_SIZE, NUM_HEADS)

    BLOCK_SIZE_K = next_power_of_2(SEQ_LEN_K)
    BLOCK_SIZE_D = next_power_of_2(D_HEAD)

    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BATCH_SIZE=BATCH_SIZE,
        SEQ_LEN_Q=SEQ_LEN_Q,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN_K=SEQ_LEN_K,
        D_HEAD=D_HEAD,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return output

# Example Usage remains the same
if __name__ == '__main__':
    BATCH_SIZE = 4
    SEQ_LEN_Q = 2048
    NUM_HEADS = 6
    SEQ_LEN_K = 6
    D_HEAD = 256

    q = torch.randn(BATCH_SIZE, SEQ_LEN_Q, NUM_HEADS, D_HEAD, device='cuda', dtype=torch.float32)
    k = torch.randn(BATCH_SIZE, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD, device='cuda', dtype=torch.float32)
    v = torch.randn(BATCH_SIZE, SEQ_LEN_Q, SEQ_LEN_K, D_HEAD, device='cuda', dtype=torch.float32)

    triton_output = mha_wrapper(q, k, v)

    def mha_pytorch_reference(q, k, v):
        scaling_factor = 1.0 / math.sqrt(q.shape[-1])
        q_ref = q.permute(0, 2, 1, 3)
        k_ref = k.unsqueeze(1)
        v_ref = v.unsqueeze(1)
        scores = torch.matmul(q_ref[:, :, :, None, :], k_ref.transpose(-2, -1)) * scaling_factor
        p = torch.softmax(scores, dim=-1)
        output = torch.matmul(p, v_ref)
        return output.squeeze(-2).permute(0, 2, 1, 3)

    pytorch_output = mha_pytorch_reference(q, k, v)

    print("Triton and PyTorch outputs are close:", torch.allclose(triton_output, pytorch_output, atol=1e-5, rtol=1e-5))
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/api_chat_4/kernel_turn4.py
Triton and PyTorch outputs are close: True
'''