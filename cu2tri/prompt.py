complex_initial_prompt = """
# CUDA to Triton Code Conversion Task

Please convert the following CUDA code to high-performance Triton code.
Current Triton version is 3.2.0. GPU is RTX 6000 Ada.

## Core Requirements

### 1. Functional Completeness
- Maintain 100% functional equivalence with the original CUDA code
- Ensure numerical accuracy and consistent computation results
- Support all original input/output shapes and data types

### 2. Performance Optimization
- Fully leverage Triton's block-level parallelism
- Optimize memory access patterns to maximize memory bandwidth utilization
- Use appropriate tile size and block size
- Avoid bank conflicts and memory access conflicts
- Reasonably use shared memory and registers

### 3. Triton Best Practices
- Use `@triton.jit` decorator with appropriate configurations
- Provide multiple tuning configurations in the triton.Config of `triton.autotune` decorator, including:
  - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K (if applicable)
  - GROUP_SIZE_M (if applicable)
  - num_warps and num_stages parameters
- Use `tl.load()` and `tl.store()` for memory operations
- Properly use `mask` parameters to handle boundary conditions
- Use `tl.dot()` for matrix multiplication (if applicable)
- Extract all tunable parameters (block sizes, tile sizes, etc.) to triton.Config decorated by @triton.autotune, pay attention to the key parameters in @triton.autotune decorator, this decorator decorates the kernel, not the wrapper or forward function
- Provide multiple configuration options to adapt to different input sizes
- Consider optimization configurations for different GPU architectures

### 4. Code Structure
- Provide a complete `forward` function as the entry point
- Clear kernel function naming (e.g., `_kernel_name`)
- Add necessary input validation and shape checking
- Include detailed docstrings explaining functionality and parameters

### 5. Error Handling and Boundary Checking
- Handle irregular tensor shapes
- Add appropriate assertion checks for input validity
- Use masks to handle boundary elements and avoid out-of-bounds access
- Remove all CUDA-specific error checking (such as cudaGetLastError, etc.)

## Original CUDA Code:
```cpp
{cuda_code}
```

## Output Requirements
Please return only the complete Python code

## Important Notes
- Ensure the generated code can run directly without additional modifications
- Code should call triton kernel through the `forward` function
- Prioritize performance, then consider code readability
- If there are multiple implementation approaches, choose the most performant one
"""
simple_initial_prompt = """
# CUDA to Triton Code Conversion Task

Please convert the following CUDA code to high-performance Triton code. Achieve the highest performance possible.
Current Triton version is 3.2.0. GPU is H100.
Please pay attention to data types.
Ensure the code entry point is the forward function, with functionality consistent with the CUDA code's forward function entry.
Please ensure the provided code is directly executable.

## Original CUDA Code:
```cpp
{cuda_code}
```

## Output Requirements
Please return only the complete Python code

"""

feedback_prompt = """The previously generated code has issues. Please fix it based on the following error information:

## Error Information
```
{error_info}
```

## Detailed Stack Trace
```
{traceback_info}
```

## Fix Requirements
1. Please carefully analyze the cause of the error
2. Fix the issues in the code
3. Ensure the code can compile and run correctly
4. Maintain functional equivalence with the CUDA code
5. Only return the complete fixed Python code, wrapped in ```python...```

Please provide the complete fixed code:"""
