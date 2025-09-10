import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, K, N,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Get the current program id for the output tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute the starting offsets for the current tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offsets for A and B tiles
        k_offs = k * BLOCK_SIZE_K + offs_k
        
        # Load A tile: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * K + k_offs[None, :])
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = b_ptr + (k_offs[:, None] * N + offs_n[None, :])
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute the partial result
        accumulator += tl.dot(a, b)
    
    # Store the result
    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_kernel(A, B, C, m, k, n):
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (tl.cdiv(m, BLOCK_SIZE_M), 
            tl.cdiv(n, BLOCK_SIZE_N))
    
    # Launch kernel
    gemm_kernel[grid](
        A, B, C, m, k, n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
