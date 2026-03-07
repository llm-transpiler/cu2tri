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
- Efficiently utilize shared memory and registers

### 3. Triton Best Practices

The Triton kernel should:
1. Import torch, triton, and triton.language as tl and other necessary modules
2. Use @triton.jit decorator on the kernel implementation (not the entrypoint function)
3. Have proper grid and block sizes
4. Boundary Handling via Masking (Implicit Padding): Use masks in tl.load/tl.store operations to handle boundary conditions of irregular input shapes. Critically, do not manually pad tensors at the Python level. The correct Triton paradigm is to pass original, un-padded tensors to the kernel's entrypoint function. The kernel's execution should be parameterized by various hardware-friendly block size constants (e.g., BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), which are managed as compile-time configurations (preferably multiples of 16, such as 16, 32, 64, 128, 256, etc.) via @triton.autotune and tl.constexpr. These constants collectively define the dimensions of the processing tile for loads and computations, not the shape of a manually created tensor within the kernel. The mask combined with other=0.0 in tl.load then performs an efficient "implicit padding" on-the-fly, which is the intended and performant method.
5. Use typed constants (tl.constexpr)
6. Handle tensor dimensions correctly
7. Return output matching CUDA's implementation
8. Extract all tunable parameters (block sizes, tile sizes, etc.) using triton.Config with @triton.autotune decorator. Pay attention to the key parameters in @triton.autotune - this decorator should decorate the kernel function, not the wrapper or forward function
9. Provide multiple configuration options to adapt to different input sizes
10. Consider optimization configurations for different GPU architectures
11. The triton.language (tl) module supports ONLY the following methods: [{tl_supported_ops_str}]. Any other methods or functions not listed here are NOT supported and must NOT be used.
12. **Template Argument Handling**: If you encounter template arguments in the CUDA code and can infer their possible values, generate either:
    - Different specialized kernels for each template value, or
    - Different configurations for the same kernel using @triton.autotune
13. Carefully verify kernel inputs and outputs, ensuring the wrapper (forward function) correctly handles data flow and maintains consistency with the original CUDA implementation.
14. **Critical Limitations**:
    - **module 'triton.language' has no attribute 'tanh'**
    - **module 'triton.language.math' has no attribute 'tanh'**
    - **module 'triton.language' has no attribute 'mul'**
    - `tl.language.dot` (`tl.dot`) constraints:
      - **Core Constraint**: tl.dot is optimized for Tensor Cores and typically requires input matrix dimensions (M, N, K) to be at least 16. Operations not meeting this requirement, especially matrix-vector multiplications, must not use tl.dot.
      - **Translation Rules**:
        - **Prohibited**: Forcing a matrix-vector product into tl.dot by reshaping the vector (e.g., (K,) -> (K, 1)). This will cause compilation errors.
          ```python
          # Incorrect: Violates dimension constraints (N=1)
          result = tl.dot(matrix, tl.reshape(vector, (K, 1)))
          ```
        - **Required**: Use broadcasting, element-wise multiplication (*), and reduction (tl.sum) for such cases.
          ```python
          # Correct: General approach, no dimension limits
          result = tl.sum(matrix * vector[None, :], axis=1)
          ```
    - The `key` parameter in `@triton.autotune` decorator specifies the kernel parameters used for autotuning. All parameters listed in `key` must be present in the decorated kernel's argument list.
    - tl.arange's arguments must be of type tl.constexpr

### 4. Code Structure
- Use clear kernel function naming (e.g., `_kernel_name`)
- Provide only the Triton kernel implementation and a `triton_kernel` function as the entry point
- Remove all input validation and shape checking from the generated code
- Include comprehensive docstrings explaining functionality and parameters

### 5. Error Handling and Boundary Checking
- Handle irregular tensor shapes gracefully
- Add appropriate assertion checks for input validity where necessary
- Use masks to handle boundary elements and prevent out-of-bounds access
- Remove all CUDA-specific error checking (such as cudaGetLastError, etc.)

## Original CUDA Code:
```cpp
{cuda_code}
```

## Output Requirements
Please return only the complete Python code

## Important Notes
- Ensure the generated code can run directly without additional modifications
- The code entry point must be the `triton_kernel` function
- Prioritize performance optimization, then consider code readability
- If multiple implementation approaches are possible, choose the most performant one
"""
simple_initial_prompt = """
# CUDA to Triton Code Conversion Task

Please convert the following CUDA code to high-performance Triton code. Achieve maximum performance optimization.
Current Triton version is 3.2.0. Target GPU: RTX 6000 Ada.

## Key Requirements
- Pay careful attention to data types and maintain consistency
- Ensure the code entry point is the `triton_kernel` function with functionality identical to the CUDA code
- Provide directly executable code without additional modifications needed

## Original CUDA Code:
```cpp
{cuda_code}
```

## Output Requirements
Return only the complete Python code implementation.
"""

feedback_prompt = """The previously generated code has encountered issues. Please fix it based on the error information provided below:

## Error Information
```
{error_info}
```

## Detailed Stack Trace
```
{traceback_info}
```

## Fix Requirements
1. Carefully analyze the root cause of the error
2. Fix all identified issues in the code
3. Ensure the code compiles and runs correctly
4. Maintain functional equivalence with the original CUDA code
5. Return only the complete fixed Python code, properly formatted within ```python...``` code blocks

Please provide the complete corrected implementation:"""

if __name__ == "__main__":
    from debugger.triton_tl.support import tl_supported_ops
    tl_supported_ops_str = ", ".join(tl_supported_ops)
    print(complex_initial_prompt.format(cuda_code="", tl_supported_ops_str=tl_supported_ops_str))