# CUDA Macro Expansion and Function Inlining Prompt

## Task
Convert CUDA kernel code containing extensive macro definitions and macro calls into a fully expanded version where all macro invocations are replaced with their actual code implementations and appropriate functions are inlined.

## Background
You will be given a CUDA kernel file that uses numerous preprocessor macros. Your task is to simulate the behavior of a C++ preprocessor, expanding all macro calls into their defined actual code while maintaining identical functionality.

## Input Format
- Original CUDA kernel file containing macro definitions and macro calls
- File includes various macros defined with #define statements

## Output Format
- Macro-expanded CUDA kernel file
- All macro definitions removed
- All macro calls replaced with actual code
- Other code structure preserved

## Expansion Strategy

### What Should Be Expanded
1. **Memory Access Macros** - Type casting and pointer manipulation macros
2. **Inline Assembly Macros** - PTX instruction wrappers (ldmatrix, mma, etc.)
3. **Utility Macros** - Simple computational or casting operations
4. **Application-Specific Constant Macros** - Non-architecture constants used as literals (exclude WARP_SIZE, etc.)

### What Should NOT Be Expanded  
1. **Complex Function-like Macros** - If they represent significant logic blocks
2. **Compiler Directives** - #pragma statements, __device__, __global__ etc.
3. **Standard CUDA/C++ Macros** - Built-in macros like __CUDA_ARCH__
4. **Configuration Macros** - Architecture-specific or build configuration macros that should remain as-is
5. **Architecture-dependent Constants** - Hardware-specific constants like WARP_SIZE that provide portability
6. **Simple Inline Functions** - Keep as functions unless specifically requested to inline

### General Expansion Rules

1. **Preserve Original Semantics** - Expansion must maintain identical functionality
2. **Remove Macro Definitions** - All #define statements should be eliminated from output  
3. **Substitute Parameters** - Macro parameters should be correctly substituted
4. **Maintain Syntax** - Expanded code must be syntactically correct C++/CUDA

## Macro Expansion Patterns

This section provides comprehensive examples of different macro expansion patterns commonly found in CUDA kernels.

### Pattern 1: Memory Access Type Casting Macros
**Purpose**: Simplify vectorized memory operations with type casting
```cpp
// Original Definitions:
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])  
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])

// Macro Usage:
LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);

// Expanded Result:
(reinterpret_cast<float4 *>(&(s_a[load_smem_a_m][load_smem_a_k]))[0]) = 
    ((reinterpret_cast<float4 *>(&(A[load_gmem_a_addr]))[0]));
```

### Pattern 2: Inline Assembly Instruction Wrappers
**Purpose**: Wrap PTX instructions in convenient macro form

#### Subpattern 2a: Matrix Load Instructions (ldmatrix)
```cpp
// Original Definitions:
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
  asm volatile( \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
      : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr) \
  asm volatile( \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
      : "=r"(R0), "=r"(R1) \
      : "r"(addr))

// Macro Usage:
LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);
LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

// Expanded Results:
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
    : "r"(load_smem_a_ptr));
    
asm volatile(
    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(RB[0]), "=r"(RB[1])
    : "r"(load_smem_b_ptr));
```

#### Subpattern 2b: Matrix Multiply-Accumulate Instructions (mma)
```cpp
// Original Definition:
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
  asm volatile( \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
      "%4, %5}, {%6, %7}, {%8, %9};\n" \
      : "=r"(RD0), "=r"(RD1) \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
        "r"(RC1))

// Macro Usage:
HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

// Expanded Result:
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "
    "%4, %5}, {%6, %7}, {%8, %9};\n"
    : "=r"(RC[0]), "=r"(RC[1])
    : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]),
      "r"(RC[1]));
```

### Pattern 3: Application-Specific Constant Definitions  
**Purpose**: Define application-specific constants (NOT architecture constants)
```cpp
// Original Definition (example of expandable constant):
#define TILE_SIZE 16

// Usage in code:
shared_memory[TILE_SIZE][TILE_SIZE];

// Expanded Result:
shared_memory[16][16];
```

**Note**: Architecture-dependent constants should NOT be expanded as they provide important portability and readability benefits. Examples include:
- `WARP_SIZE` (32 on current GPUs, but may change)  
- `MAX_THREADS_PER_BLOCK` (1024 on most current GPUs)
- `MAX_SHARED_MEMORY_PER_BLOCK` (varies by architecture)
- Any constants related to SM compute capabilities

### Pattern 4: Attribute and Qualifier Macros
**Purpose**: Standardize function attributes (Usually NOT expanded)
```cpp
// Keep as-is (do NOT expand these):
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline

// These should remain as macros for clarity and maintainability
```

## Conversion Process

### Step 1: Identify All Macro Definitions
Carefully scan the file for all `#define` statements, recording each macro's name, parameters, and definition.

### Step 2: Locate Macro Invocations  
Find all locations in the code where these macros are used.

### Step 3: Execute Macro Expansion
According to macro definitions, replace each macro call with its expanded form:
- Substitute macro parameters correctly
- Handle string concatenation if present
- Maintain proper C++/CUDA syntax

### Step 4: Remove Macro Definitions
Delete all `#define` lines from the output file.

### Step 5: Validate Syntax
Ensure the expanded code is syntactically correct.

## Important Considerations

### 1. Preserve Code Logic
- Expansion must not change the original code logic
- Maintain all variable scope and lifetime semantics  
- Preserve original computation order and data dependencies

### 2. Architecture Dependency Awareness
- **Preserve architecture-dependent constants** like `WARP_SIZE` to maintain code portability
- These constants provide self-documenting code and facilitate porting to different GPU architectures
- Expanding them would lose important semantic information and reduce maintainability

### 3. Handle Complex Macros
- Multi-line macros require proper line continuation handling
- Macros with conditional compilation need special attention
- Nested macro calls require recursive expansion

### 4. Maintain Code Style
- Preserve appropriate indentation
- Add necessary comments explaining expansion where helpful
- Keep code readable and well-formatted

### 5. Inline Assembly Handling
- Inline assembly code must maintain precise formatting
- Register constraints must remain correct
- Assembly instruction strings must not be altered

## Quality Requirements

1. **Functional Equivalence**: Expanded code must have identical functionality to original
2. **Performance Preservation**: Expansion should not introduce additional performance overhead
3. **Syntactic Correctness**: Generated code must compile successfully  
4. **Readability**: Expanded code should maintain good readability

## Validation Checklist

After conversion, verify:
1. All macro definitions have been removed
2. All macro invocations have been correctly expanded
3. Inline assembly syntax is correct
4. Variable types and scopes remain consistent
5. Code structure and logic flow are unchanged
6. No compilation errors are introduced

## Complete Example

**Before Expansion:**
```cpp
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
```

**After Expansion:**
```cpp
// Macro definitions removed

(reinterpret_cast<float4 *>(&(s_a[load_smem_a_m][load_smem_a_k]))[0]) = ((reinterpret_cast<float4 *>(&(A[load_gmem_a_addr]))[0]));
```

## Usage Instructions

Apply this macro expansion process to the provided CUDA kernel file, following all patterns and rules specified above. Focus on creating clean, readable expanded code that maintains identical functionality to the original.