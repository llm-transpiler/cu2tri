simple_initial_prompt = """
# CUDA to Ascend C Code Conversion Task

Please convert the following CUDA kernel code into Ascend C code for NPU execution.
Source stays CUDA only as translation input. The generated target must be Ascend C.

## Key Requirements
- Keep the algorithm and numerics aligned with the CUDA source
- Output must be Ascend C code (C/C++) that can be built for Ascend NPU
- Use a callable entry symbol named `ascendc_kernel` in generated code
- Do not include CUDA runtime launches in the generated target
- Keep output code self-contained and directly writable to `ascendc_/kernel.cpp`

## Original CUDA Code
```cpp
{cuda_code}
```

## Output Requirements
Return only the complete Ascend C source code in one code block.
"""


feedback_prompt = """The previously generated Ascend C code failed testing. Please fix it using the diagnostics below.

## Error Information
```
{error_info}
```

## Detailed Stderr
```
{traceback_info}
```

## Fix Requirements
1. Identify the root cause from logs and fix it in Ascend C
2. Keep functional intent equivalent to the CUDA source
3. Keep a callable symbol named `ascendc_kernel`
4. Return only the full corrected Ascend C code in a single ```cpp ... ``` block
"""
