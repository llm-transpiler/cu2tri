import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, dim1, dim2, dim3,
    stride_ab, stride_ac, stride_bb, stride_bc, stride_cb, stride_cc,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Batch Matrix Multiplication Triton kernel
    A: (batch_size, dim1, dim2)
    B: (batch_size, dim2, dim3)  
    C: (batch_size, dim1, dim3)
    """
    batch_idx = tl.program_id(axis=0)
    m_idx = tl.program_id(axis=1)
    n_idx = tl.program_id(axis=2)
    
    if batch_idx >= batch_size:
        return
    
    # Calculate offsets for this batch
    A_batch_offset = batch_idx * stride_ab
    B_batch_offset = batch_idx * stride_bb
    C_batch_offset = batch_idx * stride_cb
    
    # Calculate block ranges
    m_start = m_idx * BLOCK_SIZE_M
    n_start = n_idx * BLOCK_SIZE_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop over K dimension
    for k_start in range(0, dim2, BLOCK_SIZE_K):
        # Load A block
        a_offs = (A_batch_offset + 
                 (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_ac +
                 (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :])
        a_mask = ((m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < dim1) & \
                 ((k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] < dim2)
        a_vals = tl.load(A_ptr + a_offs, mask=a_mask, other=0.0)
        
        # Load B block  
        b_offs = (B_batch_offset +
                 (k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bc +
                 (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :])
        b_mask = ((k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] < dim2) & \
                 ((n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < dim3)
        b_vals = tl.load(B_ptr + b_offs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a_vals, b_vals)
    
    # Store result
    c_offs = (C_batch_offset +
             (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cc +
             (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :])
    c_mask = ((m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < dim1) & \
             ((n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < dim3)
    tl.store(C_ptr + c_offs, acc, mask=c_mask)


def triton_kernel(A, B):
    """
    Triton BMM implementation
    """
    batch_size, dim1, dim2 = A.shape
    _, _, dim3 = B.shape
    
    C = torch.empty((batch_size, dim1, dim3), dtype=A.dtype, device=A.device)
    
    # Calculate strides
    stride_ab, stride_ac = A.stride(0), A.stride(1) 
    stride_bb, stride_bc = B.stride(0), B.stride(1)
    stride_cb, stride_cc = C.stride(0), C.stride(1)
    
    # Grid dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = (
        batch_size,
        triton.cdiv(dim1, BLOCK_SIZE_M),
        triton.cdiv(dim3, BLOCK_SIZE_N)
    )
    
    bmm_kernel[grid](
        A, B, C,
        batch_size, dim1, dim2, dim3,
        stride_ab, stride_ac, stride_bb, stride_bc, stride_cb, stride_cc,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C