import torch
import triton
import triton.language as tl
import math

# Triton Kernel (Corrected Version 2)
@triton.jit
def mha_triton_kernel(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    # Stride information for correct memory access
    stride_q_b, stride_q_s, stride_q_h,
    stride_k_b, stride_k_s, stride_k_h,
    stride_v_b, stride_v_s, stride_v_h,
    stride_o_b, stride_o_s, stride_o_h,
    # Other parameters
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    """
    Triton kernel for a non-standard "cross-head" attention mechanism.
    Each instance of this kernel computes the attention for one item in the batch
    and one position in the sequence.

    Grid: (BATCH_SIZE, SEQ_LEN)
    """
    # 1. Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # 2. Create block pointers
    q_start_ptr = Q_ptr + pid_batch * stride_q_b + pid_seq * stride_q_s
    k_start_ptr = K_ptr + pid_batch * stride_k_b + pid_seq * stride_k_s
    v_start_ptr = V_ptr + pid_batch * stride_v_b + pid_seq * stride_v_s

    # --- FIX IS HERE ---
    # Instead of creating a BLOCK_SHAPE variable, pass the tuple of constexpr
    # values directly into the function call.

    # Create a block pointer for Q
    q_block_ptr = tl.make_block_ptr(
        base=q_start_ptr,
        shape=(NUM_HEADS, HEAD_DIM),
        strides=(stride_q_h, 1),
        offsets=(0, 0),
        block_shape=(NUM_HEADS, HEAD_DIM), # Must be a literal tuple of constants
        order=(1, 0)
    )
    # Create a block pointer for K
    k_block_ptr = tl.make_block_ptr(
        base=k_start_ptr,
        shape=(NUM_HEADS, HEAD_DIM),
        strides=(stride_k_h, 1),
        offsets=(0, 0),
        block_shape=(NUM_HEADS, HEAD_DIM), # Must be a literal tuple of constants
        order=(1, 0)
    )
    # Create a block pointer for V
    v_block_ptr = tl.make_block_ptr(
        base=v_start_ptr,
        shape=(NUM_HEADS, HEAD_DIM),
        strides=(stride_v_h, 1),
        offsets=(0, 0),
        block_shape=(NUM_HEADS, HEAD_DIM), # Must be a literal tuple of constants
        order=(1, 0)
    )

    # Load the data into SRAM
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # 3. Compute Q @ K.T
    scores = tl.dot(q, tl.trans(k))

    # 4. Apply scaling factor
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    scores = scores * scaling_factor

    # 5. Apply softmax
    attn_weights = tl.softmax(scores, axis=1)
    attn_weights = attn_weights.to(tl.float32)

    # 6. Compute Softmax @ V
    output = tl.dot(attn_weights, v)

    # 7. Store the result back to global memory
    output_start_ptr = output_ptr + pid_batch * stride_o_b + pid_seq * stride_o_s
    output_block_ptr = tl.make_block_ptr(
        base=output_start_ptr,
        shape=(NUM_HEADS, HEAD_DIM),
        strides=(stride_o_h, 1),
        offsets=(0, 0),
        block_shape=(NUM_HEADS, HEAD_DIM), # Must be a literal tuple of constants
        order=(1, 0)
    )
    tl.store(output_block_ptr, output)


# Wrapper function and verification script remain unchanged.
def mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    output = torch.empty_like(q)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    grid = (BATCH, SEQ_LEN)

    mha_triton_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        SEQ_LEN=SEQ_LEN,
        NUM_HEADS=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
    )
    return output

def torch_reference_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    output = torch.zeros_like(q)
    scaling_factor = 1.0 / math.sqrt(HEAD_DIM)
    
    for b in range(BATCH):
        for s in range(SEQ_LEN):
            q_slice = q[b, s, :, :]
            k_slice = k[b, s, :, :]
            v_slice = v[b, s, :, :]
            
            scores = torch.matmul(q_slice, k_slice.T)
            scores *= scaling_factor
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            output_slice = torch.matmul(attn_weights, v_slice)
            output[b, s, :, :] = output_slice
            
    return output

if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    NUM_HEADS = 6
    HEAD_DIM = 256

    q = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    k = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)
    v = torch.randn((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), device='cuda', dtype=torch.float32)

    triton_output = mha(q, k, v)

    print("Running verification on a smaller tensor size to avoid long wait times...")
    q_small = q[:, :4, :, :].clone()
    k_small = k[:, :4, :, :].clone()
    v_small = v[:, :4, :, :].clone()
    
    triton_output_small = mha(q_small, k_small, v_small)
    torch_output_small = torch_reference_mha(q_small, k_small, v_small)

    print(f"Triton output shape: {triton_output.shape}")
    if torch.allclose(triton_output_small, torch_output_small, atol=1e-5, rtol=1e-4):
        print("✅ Triton and PyTorch results match!")
    else:
        print("❌ Triton and PyTorch results DO NOT match!")
        print("Max difference:", (triton_output_small - torch_output_small).abs().max().item())
'''
root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/mha/triton/kernel.py
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 2231, in make_block_ptr
    return _semantic.make_block_ptr(base, shape, strides, offsets, block_shape, order)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 1838, in make_block_ptr
    return self.tensor(handle, tl.pointer_type(tl.block_type(base.type.element_ty, block_shape)))
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 726, in __init__
    self.numel = validate_block_shape(self.shape)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/_utils.py", line 52, in validate_block_shape
    raise ValueError(f"Shape element {i} must be a power of 2")
ValueError: Shape element 0 must be a power of 2

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 150, in <module>
    triton_output = mha(q, k, v)
                    ^^^^^^^^^^^^
  File "/workspace/cu2til/cases/mha/triton/kernel.py", line 109, in mha
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
triton.compiler.errors.CompilationError: at 36:18:

    # 2. Create block pointers
    q_start_ptr = Q_ptr + pid_batch * stride_q_b + pid_seq * stride_q_s
    k_start_ptr = K_ptr + pid_batch * stride_k_b + pid_seq * stride_k_s
    v_start_ptr = V_ptr + pid_batch * stride_v_b + pid_seq * stride_v_s

    # --- FIX IS HERE ---
    # Instead of creating a BLOCK_SHAPE variable, pass the tuple of constexpr
    # values directly into the function call.

    # Create a block pointer for Q
    q_block_ptr = tl.make_block_ptr(
                  ^
'''