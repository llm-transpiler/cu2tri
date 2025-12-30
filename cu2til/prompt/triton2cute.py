
from cu2til.llm_trans.utils.gpu_targets import get_gpu_target, format_prompt_hint

complex_initial_prompt = """
# Triton → CUTE (CUTLASS 3.x) Conversion Task

Target GPU: {gpu_label} ({gpu_arch}, {gpu_memory})
CUTLASS architecture: {cutlass_arch}
{gpu_guidance}

## Core Requirements

### 1. Functional Completeness
- Maintain 100% functional equivalence with the original Triton code
- Ensure numerical accuracy and consistent computation results
- Support all original input/output shapes and data types

### 2. Performance Optimization
- Fully leverage CUTE's tile abstraction and threadblock scheduling
- Use proper Layout and Tensor types for optimal memory access patterns
- Optimize for tensor core operations when applicable
- Efficiently utilize shared memory and registers through CUTE's Copy atoms
- Use appropriate tile scheduling and pipelining

### 3. CUTE/CUTLASS Best Practices

The CUTE kernel should:
1. Include necessary CUTLASS/CUTE headers:
   ```cpp
   #include <cute/tensor.hpp>
   #include <cutlass/cutlass.h>
   #include <cutlass/numeric_types.h>
   // Other necessary headers
   ```
2. Use CUTE's Layout and Tensor types for memory management
3. Implement proper thread block decomposition using CUTE's tile abstractions
4. Use Copy atoms for efficient data movement
5. Leverage MMA atoms for tensor core operations when applicable
6. Handle boundary conditions properly with predicates
7. Use proper synchronization mechanisms
8. Implement efficient shared memory usage patterns
9. Return output matching Triton's implementation
10. Consider optimization for different GPU architectures (SM80, SM89, SM90)

### 4. Code Structure
- Provide a complete CUDA kernel function
- **IMPORTANT**: Include a host-side wrapper function with C linkage:
  ```cpp
  extern "C" {{
      void cute_kernel_wrapper(float* a, float* b, float* output, int M, int N);
  }};
  ```
  The function signature must match the expected interface from the test harness.
  Adjust parameter types based on the actual data types used.
- Use clear naming conventions
- Include comprehensive comments explaining functionality and parameters
- Handle grid and block dimensions appropriately

### 5. Error Handling and Boundary Checking
- Handle irregular tensor shapes gracefully
- Add appropriate boundary checks
- Use predicates for conditional loads/stores
- Include CUDA error checking where appropriate

## Original Triton Code
```python
{triton_code}
```

## Additional Context
- Input shapes: {input_shapes}
- Data types: {data_types}

## Output Rules (strict)
- Output a single Markdown code block containing the complete C++ implementation
- The code block must start with three backticks + cpp and end with three backticks
- Include both kernel implementation and host wrapper function with `extern "C"` linkage
- The wrapper function must be named `cute_kernel_wrapper` for the test harness to load it
- Do not output any text or blank lines outside the code block
- If you must include a literal sequence of three backticks inside comments, construct them at runtime or use different comment styles

## Important Notes
- The generated code should compile with nvcc and CUTLASS 3.x headers
- Include proper CUDA/CUTLASS error checking
- Prioritize performance optimization while maintaining code readability
- Use CUTE's abstractions (Layout, Tensor, Copy, MMA) appropriately
- Consider memory coalescing and bank conflict avoidance
"""

simple_initial_prompt = """
# Triton → CUTE Quick Conversion

Target GPU: {gpu_label} ({gpu_arch}, {gpu_memory})

Requirements:
- Keep functional parity and numerics identical to Triton reference
- Use CUTE Layout/Tensor primitives with CUTLASS {cutlass_arch} kernels
- Emit kernel + `extern "C"` wrapper inside one ```cpp block

Source:
```python
{triton_code}
```
"""


def _prompt_tokens(target_gpu: str) -> dict[str, str]:
    spec = get_gpu_target(target_gpu)
    return {
        "gpu_label": spec.label,
        "gpu_arch": spec.architecture,
        "gpu_memory": spec.memory,
        "cutlass_arch": spec.cutlass_arch,
        "gpu_guidance": format_prompt_hint(spec),
    }


def get_initial_prompt(
    triton_code: str,
    use_simple: bool = False,
    *,
    target_gpu: str = "h800_sxm",
    **kwargs,
) -> str:
    """
    Generate initial prompt for Triton to CUTE conversion.
    
    Args:
        triton_code: The Triton source code to convert
        use_simple: If True, use simple prompt template
        **kwargs: Additional context (input_shapes, data_types, etc.)
    
    Returns:
        Formatted prompt string
    """
    template = simple_initial_prompt if use_simple else complex_initial_prompt
    tokens = _prompt_tokens(target_gpu)
    input_shapes = kwargs.get("input_shapes", "Not specified")
    data_types = kwargs.get("data_types", "Not specified")

    return template.format(
        triton_code=triton_code,
        input_shapes=input_shapes,
        data_types=data_types,
        **tokens,
    )


feedback_prompt = """
# Compilation/Test Error Feedback

The generated CUTE code has the following issues:

## Error Output:
```
{error_output}
```

## Current Code:
```cpp
{current_code}
```

Please fix the code to resolve these errors. Key points:
1. Check CUTE/CUTLASS API usage
2. Verify Layout and Tensor type definitions
3. Ensure proper boundary handling
4. Fix any compilation errors
5. Maintain functional correctness

Output the fixed code in a single ```cpp code block.
"""

def get_feedback_prompt(error_output: str, current_code: str) -> str:
    """
    Generate feedback prompt for error correction.
    
    Args:
        error_output: Compilation or test error messages
        current_code: Current CUTE code that failed
    
    Returns:
        Formatted feedback prompt
    """
    return feedback_prompt.format(
        error_output=error_output,
        current_code=current_code
    )
