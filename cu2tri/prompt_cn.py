complex_initial_prompt = """
# CUDA到Triton代码转换任务

请将以下CUDA代码转换为高性能的Triton代码。
现在的triton版本为3.2.0。GPU是H100

## 核心要求

### 1. 功能完整性
- 保持与原CUDA代码100%功能等价
- 确保数值精度和计算结果一致
- 支持所有原始输入输出形状和数据类型

### 2. 性能优化
- 充分利用Triton的block-level parallelism
- 优化内存访问模式，最大化内存带宽利用率
- 使用适当的tile size和block size
- 避免bank conflicts和内存访问冲突
- 合理使用共享内存和寄存器

### 3. Triton最佳实践
- 使用`@triton.jit`装饰器和适当的配置
- 提供多个tuning配置在`triton.autotune`装饰器的triton.Config中，包括：
  - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K（如适用）
  - GROUP_SIZE_M（如适用）
  - num_warps和num_stages参数
- 使用`tl.load()`和`tl.store()`进行内存操作
- 合理使用`mask`参数处理边界条件
- 使用`tl.dot()`进行矩阵乘法（如适用）
- 将所有可调参数（block sizes, tile sizes等）提取到@triton.autotune装饰的triton.Config中，注意@triton.autotune装饰器中key的参数，这个装饰器装饰的是kernel，而不是wrapper或者forward函数
- 提供多个配置选项以适应不同的输入大小
- 考虑不同GPU架构的优化配置

### 4. 代码结构
- 提供完整的`forward`函数作为入口点
- 内核函数命名清晰（如`_kernel_name`）
- 添加必要的输入验证和形状检查
- 包含详细的docstring说明功能和参数

### 5. 错误处理和边界检查
- 处理不规则的tensor shape
- 添加适当的断言检查输入有效性
- 使用mask处理边界元素，避免越界访问
- 移除所有CUDA特有的错误检查（如cudaGetLastError等）

## 原始CUDA代码：
```cpp
{cuda_code}
```

## 输出要求
请只返回完整的Python代码

## 重要提示
- 确保生成的代码可以直接运行，无需额外修改
- 代码应该通过`forward`函数调用triton kernel
- 优先考虑性能，其次考虑代码可读性
- 如有多种实现方案，选择性能最优的方案
"""
simple_initial_prompt = """
# CUDA到Triton代码转换任务

请将以下CUDA代码转换为高性能的Triton代码。性能尽可能高。
现在的triton版本为3.2.0。GPU是H100。
请注意数据类型。
保证代码的入口为forward函数，和CUDA代码的forward函数入口功能一致。
请保证给出的代码是直接可以运行的。

## 原始CUDA代码：
```cpp
{cuda_code}
```

## 输出要求
请只返回完整的Python代码

"""

feedback_prompt = """上一次生成的代码存在问题，请根据以下错误信息进行修复：

## 错误信息
```
{error_info}
```

## 详细堆栈信息
```
{traceback_info}
```

## 修复要求
1. 请仔细分析错误原因
2. 修复代码中的问题
3. 确保代码可以正确编译和运行
4. 保持与CUDA代码的功能等价性
5. 只返回修复后的完整Python代码，用```python...```包裹

请提供修复后的完整代码："""
