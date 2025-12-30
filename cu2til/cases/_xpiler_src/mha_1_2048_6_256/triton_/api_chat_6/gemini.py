from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="totemic-fact-469314-n2",
      location="global",
  )

  msg1_text1 = types.Part.from_text(text="""__global__ void kernel(float *Q, float *K, float *V, float *output) {

  __shared__ float score[36]; // 使用共享内存存储 score, 大小为 heads * heads
  float scaling_factor = 1.0f / sqrtf((float)256);
  int i = blockIdx.x;  // batch index
  int j = blockIdx.y;  // query index within sequence length
  int m = threadIdx.x; // head index

  for (int n = 0; n < 6; n++) {
    score[m * 6 + n] = 0.0;
    for (int p = 0; p < 256; p++) {
      score[m * 6 + n] += Q[i * 2048 * 6 * 256 + j * 6 * 256 + m * 256 + p] *
                          K[i * 2048 * 6 * 256 + j * 6 * 256 + n * 256 + p];
    }
  }
  // score
  for (int n_sc = 0; n_sc < 6; n_sc++) {
    score[m * 6 + n_sc] = score[m * 6 + n_sc] * scaling_factor;
  }

  float sum = 0;

  for (int i_ex = 0; i_ex < 6; ++i_ex) {
    score[m * 6 + i_ex] = expf(score[m * 6 + i_ex]);
  }
  for (int i_sf = 0; i_sf < 6; ++i_sf) {
    sum += score[m * 6 + i_sf];
  }
  for (int k_sf = 0; k_sf < 6; ++k_sf) {
    score[m * 6 + k_sf] = score[m * 6 + k_sf] / sum;
  }
  // The final Matmul
  for (int n_fl = 0; n_fl < 256; ++n_fl) {
    output[i * 2048 * 6 * 256 + j * 6 * 256 + m * 256 + n_fl] = 0.0;
    for (int k_fl = 0; k_fl < 6; ++k_fl) {
      output[i * 2048 * 6 * 256 + j * 6 * 256 + m * 256 + n_fl] +=
          score[m * 6 + k_fl] *
          V[i * 2048 * 6 * 256 + j * 6 * 256 + k_fl * 256 + n_fl];
    }
  }
}

extern \"C\" void cuda_kernel(float *d_queries, float *d_keys, float *d_values, float *d_output) {
  const int batch_size = 1;
  const int seq_len = 2048;
  const int num_heads = 6;
  const int head_dim = 256;
  dim3 grid(batch_size, seq_len);
  dim3 block(num_heads);

  kernel<<<grid, block>>>(d_queries, d_keys, d_values, d_output);
}


Please translate it to triton""")
  msg2_text1 = types.Part.from_text(text="""**Understanding the Kernel's Function**

I've begun to unravel the CUDA kernel. Initially, I identified its self-attention nature, though a unique implementation. Now, I'm focused on dissecting the code block by block, aiming to grasp the specific operations and data flow for efficient Triton translation.


**Decomposing the Core Logic**

I'm carefully examining the kernel's internal loop. Specifically, the shared memory usage for `score`. I'm also tracing data movement between global memory (Q, K) and shared memory, paying attention to how the scaling factor and the matrix multiplication `Q @ K^T` influence the final output.


**Dissecting Intra-Token Head Attention**

The current focus is to confirm the unusual nature of this kernel: intra-token head attention, not standard self-attention. It's attending within each token across heads, which simplifies the computation but suggests a different use case. The shared memory allocation seems optimized for this structure, pointing toward further efficiency gains.


**Clarifying Intra-Token Operations**

I'm working on fully understanding the peculiar \"intra-token head attention.\" It involves a (6x6) score matrix per token, calculated via a dot product of heads within that token. The softmax is applied row-wise. This operation suggests highly specific input/output relationships, which will influence Triton's implementation.


**Formalizing Tensor Operations**

I'm now crystallizing the exact tensor operations. Specifically, I'm confirming the unusual attention mechanism involves a `[6x6]` attention matrix for each token, computed with an inner product between heads. Then, a row-wise softmax on this matrix followed by a matrix multiplication with the `V` matrix. This clarifies the data flow and will drive Triton's kernel design.


**Defining Triton's Structure**

I'm now designing the Triton kernel structure. The 2D grid maps directly from CUDA. Based on the small head dimension, I plan to process all heads within a Triton program. Block sizes are being chosen for optimal performance, considering the unusual matrix multiplications involved in the intra-token head attention mechanism.


**Defining Program Parameters**

I'm now defining the best approach for a Triton implementation, opting for efficient loading and processing within each program instance, driven by the limited size of each tensor slice (`6x256`). The key is to make sure we can load the `Q`, `K`, and `V` slices into registers or shared memory. The overall design will be driven by this key observation.


**Formalizing Data Access**

I'm now focused on the access patterns for `Q`, `K`, and `V`. Considering the `intra-token` nature, each program instance needs to load a `[6, 256]` slice from each tensor. The critical aspect is efficient loading to optimize memory bandwidth, especially for shared memory usage and register allocation.


**Revisiting Data Loading**

I'm re-evaluating the data loading process for `Q`, `K`, and `V` tensors. The goal is to minimize memory traffic, prioritizing shared memory or register usage, especially given the intra-token head attention. I must ensure the code's efficiency matches the design.


**Simplifying the Implementation**

I'm now simplifying the Triton kernel implementation. The small tensor dimensions lead to an approach where loading and processing small tensor slices in SRAM is sufficient. This avoids unnecessary complexity. I'm choosing an implementation that directly loads the `Q`, `K`, and `V` slices and directly performs the necessary calculations.


**Simplifying the Design**

I'm now focused on simplifying the entire implementation. Given the small slice sizes and the intra-token head attention design, I will load small tensors into registers, minimizing shared memory use. The critical step is to directly perform the calculations within the Triton kernel.


**Simplifying Implementation Further**

I'm now focused on simplifying the Triton kernel. Given the small slice sizes and the intra-token head attention design, I'll load `Q`, `K`, and `V` slices into registers, and directly perform the calculations within the Triton kernel, without shared memory.


**Optimizing Memory Loading**

I'm now focused on optimizing the access patterns for the tensors, `Q`, `K`, and `V`. Given the small slice dimensions of each tensor (6, 256), efficient loading is critical. I'm considering register usage and the possibility of shared memory to optimize bandwidth.


**Streamlining Kernel Structure**

I'm now analyzing the kernel, with a focus on optimizing the logic and reducing complexity. My goal is to keep the implementation simple, especially regarding memory access patterns and computations. The key will be the selection of block sizes for efficient data loading and kernel execution.


**Finalizing Kernel Design**

I'm now building the final Triton kernel. The memory layout is crucial; I'm optimizing the loads of `Q`, `K`, and `V`, and performing the calculation in SRAM using register-level loads. I'm focusing on simplifying the logic while making sure the execution is optimized.""")
  msg2_text2 = types.Part.from_text(text="""Of course. Here is the equivalent implementation of your CUDA kernel in Triton.

### Analysis of the CUDA Kernel

First, let's break down what the original CUDA kernel does. It performs a self-attention operation, but with a very specific structure:

1.  **Grid/Block Dimensions:**
    *   The grid is `(batch_size, seq_len)`. This means each CUDA block is responsible for computing the output for a **single token** in the sequence.
    *   The block size is `num_heads`. This means each thread within a block is responsible for a **single query head**.

2.  **Core Logic (for a single token at `batch_idx, seq_idx`):**
    *   It takes the `Q`, `K`, and `V` slices corresponding to that single token: `Q[batch_idx, seq_idx, :, :]`, `K[batch_idx, seq_idx, :, :]`, `V[batch_idx, seq_idx, :, :]`. Each of these slices has the shape `[num_heads, head_dim]`.
    *   **Matmul 1:** It computes an attention score matrix `S` by performing a matrix multiplication between the query heads and the key heads *at the same token position*: `S = Q_slice @ K_slice.T`. The resulting `S` matrix has the shape `[num_heads, num_heads]`.
    *   **Softmax:** It applies scaling and a row-wise softmax to the `S` matrix.
    *   **Matmul 2:** It computes the final output by multiplying the softmax-scaled scores `P` with the value heads: `Output_slice = P @ V_slice`. The result has the shape `[num_heads, head_dim]`.
    *   This entire process is repeated for every token in the sequence across the different CUDA blocks.

### Triton Implementation Strategy

Triton is perfectly suited for this kind of fused kernel. We will design a Triton kernel that mirrors the CUDA block's responsibility: one Triton program instance will handle one token.

1.  **Program IDs:** We'll use `tl.program_id(0)` for the batch index and `tl.program_id(1)` for the sequence index.
2.  **Data Loading:** For each program instance, we will load the entire `[num_heads, head_dim]` slices of `Q`, `K`, and `V` for the corresponding token into SRAM.
3.  **Computation:** We will use `tl.dot` for the two matrix multiplications and Triton's element-wise operations for the scaling and softmax. This fuses the entire calculation into a single kernel, minimizing memory traffic to and from global memory.
4.  **Data Storing:** The final `[num_heads, head_dim]` output block is written back to global memory.

### Translated Triton Code

Here is the complete, commented Triton code along with a Python launcher function.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def attention_kernel_triton(
    # Pointers to tensors
    Q_ptr, K_ptr, V_ptr, output_ptr,
    # Stride variables for memory access
    stride_q_bs, stride_q_sl, stride_q_h, stride_q_d,
    stride_k_bs, stride_k_sl, stride_k_h, stride_k_d,
    stride_v_bs, stride_v_sl, stride_v_h, stride_v_d,
    stride_o_bs, stride_o_sl, stride_o_h, stride_o_d,
    # Matrix dimensions
    num_heads,
    head_dim,
    # Triton block sizes (powers of 2 for performance)
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"
    Triton kernel for intra-token head attention.
    Each program instance computes the attention for a single token across all its heads.
    \"\"\"
    # 1. Get program IDs to identify the current token
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # 2. Compute pointers to the start of the Q, K, V slices for the current token
    q_start_ptr = Q_ptr + batch_idx * stride_q_bs + seq_idx * stride_q_sl
    k_start_ptr = K_ptr + batch_idx * stride_k_bs + seq_idx * stride_k_sl
    v_start_ptr = V_ptr + batch_idx * stride_v_bs + seq_idx * stride_v_sl

    # 3. Create ranges and masks for loading data blocks
    # Ranges for the head and dimension axes
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Create 2D blocks of pointers for Q and V
    q_ptrs = q_start_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
    v_ptrs = v_start_ptr + offs_h[:, None] * stride_v_h + offs_d[None, :] * stride_v_d
    
    # Create a transposed 2D block of pointers for K to load K.T directly
    k_t_ptrs = k_start_ptr + offs_d[:, None] * stride_k_d + offs_h[None, :] * stride_k_h

    # Create masks to prevent out-of-bounds access if dimensions are not powers of 2
    mask_h = offs_h < num_heads
    mask_d = offs_d < head_dim # Not strictly needed if BLOCK_D == head_dim

    # 4. Load Q, K.T, and V slices from HBM into SRAM
    # Load Q slice [num_heads, head_dim]
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    # Load K.T slice [head_dim, num_heads]
    k_t = tl.load(k_t_ptrs, mask=mask_h[None, :] & mask_d[:, None], other=0.0)
    # Load V slice [num_heads, head_dim]
    v = tl.load(v_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # 5. Compute S = Q @ K.T
    # The result 'scores' will have shape [BLOCK_H, BLOCK_H]
    scores = tl.dot(q, k_t)

    # 6. Apply scaling and softmax
    scaling_factor = (1.0 / tl.sqrt(head_dim.to(tl.float32)))
    scores *= scaling_factor

    # Create a mask for the score matrix to handle padding in the head dimension
    score_mask = mask_h[:, None] & mask_h[None, :]
    # Set scores for padded heads to -inf before softmax
    scores = tl.where(score_mask, scores, -float('inf'))

    # Compute softmax row-wise
    row_max = tl.max(scores, axis=1)
    scores -= row_max[:, None]  # For numerical stability
    numerator = tl.exp(scores)
    denominator = tl.sum(numerator, axis=1)
    softmax_scores = numerator / denominator[:, None]
    
    # Zero out any NaNs that might have resulted from empty rows
    softmax_scores = tl.where(score_mask, softmax_scores, 0.0)

    # 7. Compute final output O = P @ V
    # P is [num_heads, num_heads], V is [num_heads, head_dim]
    # The result 'output' will have shape [BLOCK_H, BLOCK_D]
    # Ensure dtypes match for tl.dot
    output = tl.dot(softmax_scores.to(v.dtype), v)

    # 8. Write the output block back to HBM
    output_start_ptr = output_ptr + batch_idx * stride_o_bs + seq_idx * stride_o_sl
    output_ptrs = output_start_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :] * stride_o_d
    output_mask = mask_h[:, None] & mask_d[None, :]
    tl.store(output_ptrs, output, mask=output_mask)


def triton_kernel_launcher(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    \"\"\"
    Python launcher for the Triton attention kernel.
    \"\"\"
    # Ensure inputs are on the correct device and have the expected layout
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    
    # Get tensor dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Create an empty output tensor
    output = torch.empty_like(q)
    
    # Define the grid for the kernel launch
    # One program per token in the batch
    grid = (batch_size, seq_len)

    # Define Triton block sizes. They should be powers of 2 for best performance.
    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Launch the kernel
    attention_kernel_triton[grid](
        # Tensors
        q, k, v, output,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for Output
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Dimensions
        num_heads,
        head_dim,
        # Block sizes
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    
    return output

# --- Example Usage and Verification ---
if __name__ == '__main__':
    # Parameters from the CUDA kernel
    batch_size = 1
    seq_len = 2048
    num_heads = 6
    head_dim = 256

    # Create random input tensors
    torch.manual_seed(0)
    q = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    k = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)
    v = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda', dtype=torch.float32)

    # Run the Triton kernel
    triton_output = triton_kernel_launcher(q, k, v)

    # --- PyTorch reference implementation for verification ---
    # This performs the same logic as the CUDA/Triton kernel
    import math
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    p = torch.nn.functional.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(p, v)
    
    # Compare results
    print(\"Triton vs. PyTorch allclose:\", torch.allclose(triton_output, pytorch_output, atol=1e-5, rtol=1e-4))
    
    # Print a small slice to visually inspect
    print(\"\
Triton output (slice):\")
    print(triton_output[0, 0, :3, :5])
    
    print(\"\
PyTorch output (slice):\")
    print(pytorch_output[0, 0, :3, :5])

```

### Key Differences and Advantages of the Triton Version

1.  **Conciseness and Readability:** The Triton kernel is significantly shorter and expresses the high-level mathematical operations (`tl.dot`, `tl.exp`, `tl.sum`) directly, rather than through explicit C++ loops.
2.  **Performance:** Triton's compiler is highly optimized for this type of workload. It automatically handles shared memory management, instruction scheduling, and memory coalescing, which often leads to performance comparable to or better than hand-tuned CUDA.
3.  **Fusion:** The entire operation (Matmul -> Scale -> Softmax -> Matmul) is fused into a single kernel. This is a major performance win as it keeps intermediate data (`scores`, `softmax_scores`) in fast SRAM instead of writing it back to and reading it from slow global memory (HBM).
4.  **Abstraction:** You work with blocks of data (`q`, `k_t`, `v`) rather than individual elements. This higher level of abstraction makes the code easier to write and maintain.
5.  **Portability:** The Python-based launcher makes it easy to integrate into modern deep learning frameworks like PyTorch.""")
  msg3_text1 = types.Part.from_text(text="""root@ubuntu-ThinkStation-P520:/workspace# /usr/bin/python /workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py
Traceback (most recent call last):
  File \"/usr/local/lib/python3.12/dist-packages/triton/language/core.py\", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.12/dist-packages/triton/language/core.py\", line 2045, in dot
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py\", line 1503, in dot
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \\
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File \"/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py\", line 153, in <module>
    triton_output = triton_kernel_launcher(q, k, v)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File \"/workspace/cu2til/cases/xpiler/mha_1_2048_6_256/triton_/api_chat_6/kernel_turn1.py\", line 117, in triton_kernel_launcher
    attention_kernel_triton[grid](
  File \"/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py\", line 390, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py\", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py\", line 339, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File \"/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py\", line 83, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 55:13:

    # 4. Load Q, K.T, and V slices from HBM into SRAM
    # Load Q slice [num_heads, head_dim]
    q = tl.load(q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    # Load K.T slice [head_dim, num_heads]
    k_t = tl.load(k_t_ptrs, mask=mask_h[None, :] & mask_d[:, None], other=0.0)
    # Load V slice [num_heads, head_dim]
    v = tl.load(v_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # 5. Compute S = Q @ K.T
    # The result 'scores' will have shape [BLOCK_H, BLOCK_H]
    scores = tl.dot(q, k_t)
             ^""")

  model = "gemini-2.5-pro"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
    types.Content(
      role="model",
      parts=[
        msg2_text1,
        msg2_text2
      ]
    ),
    types.Content(
      role="user",
      parts=[
        msg3_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.35,
    top_p = 0.95,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()