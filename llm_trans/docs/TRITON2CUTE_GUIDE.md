# Triton 到 CUTE 翻译系统使用指南

本文档介绍如何使用扩展后的系统进行 Triton 到 CUTE (CUTLASS 3.x) 的代码翻译。

## 系统概述

系统现在支持多种翻译方向：
- **CUDA → Triton** (原有功能)
- **Triton → CUTE/CUTLASS** (新功能)

## 新增功能

### 1. 翻译方向配置

通过新增的 CLI 参数控制：

```bash
--source-lang {cuda,triton}  # 源语言 (默认: cuda)
--target-lang {triton,cute,cutlass}  # 目标语言 (默认: triton)
```

### 2. 新的测试集

添加了两个 Triton→CUTE 专用测试集：
- `triton2cute_add`: 简单的向量加法测试
- `triton2cute_fa`: Flash Attention 测试

### 3. CUTE 编译和测试框架

- 自动编译 C++ CUTE 代码
- 与 PyTorch 参考实现对比验证正确性
- 支持性能基准测试

## 快速开始

### 环境准备

1. **设置 CUTLASS 路径**（如果不在默认位置）：
```bash
export CUTLASS_ROOT=/data/apps/project/cu2tri/elib/cutlass_latest
```

2. **激活 conda 环境**：
```bash
conda activate serve
```

3. **确保 GPU 可用**：
```bash
export CUDA_VISIBLE_DEVICES=7  # 或其他可用 GPU
```

### 基本使用

#### 测试 Add 算子翻译

```bash
python -m cu2til.llm_trans \
  --model gpt_5_mini \
  --testset triton2cute_add \
  --source-lang triton \
  --target-lang cute \
  --max-rounds 3 \
  --max-attempts 1 \
  --concurrency 1
```

#### 测试 Flash Attention 翻译

```bash
python -m cu2til.llm_trans \
  --model gpt_5_mini \
  --testset triton2cute_fa \
  --source-lang triton \
  --target-lang cute \
  --max-rounds 5 \
  --max-attempts 1 \
  --concurrency 1
```

## 测试用例结构

每个 Triton→CUTE 测试用例包含以下文件：

```
add_simple/
├── triton_/
│   └── kernel.py          # Triton 源代码
├── torch_/
│   └── ref.py            # PyTorch 参考实现
├── cute_/
│   ├── kernel.cu         # CUTE 目标代码（由 LLM 生成）
│   └── kernel_template.cu # CUTE 实现模板（可选）
├── get_data.py           # 测试数据生成
├── check_cute.py         # CUTE 测试脚本
└── logs/                 # 测试日志目录
```

## 工作流程

1. **读取 Triton 源代码**
   - 系统从 `triton_/kernel.py` 读取 Triton 实现

2. **LLM 翻译**
   - 使用 `cu2til/prompt/triton2cute.py` 中的 prompt
   - LLM 生成 CUTE C++ 代码

3. **保存并编译**
   - 将生成的代码保存到 `cute_/kernel.cu`
   - 使用 nvcc 编译成共享库

4. **测试验证**
   - 运行 `check_cute.py` 进行正确性测试
   - 与 PyTorch 参考实现对比

5. **迭代修复**
   - 如果编译或测试失败，将错误反馈给 LLM
   - LLM 生成修复版本
   - 重复步骤 3-4 直到成功或达到最大轮数

## Prompt 模板

### 主 Prompt (triton2cute.py)

包含以下关键指导：
- CUTE/CUTLASS 3.x 基础知识
- Layout 和 Tensor 抽象
- Copy atoms 和 MMA atoms
- 性能优化建议
- 边界处理
- 示例代码结构

### 反馈 Prompt

用于编译/测试错误的修复：
- 提供完整错误信息
- 当前代码上下文
- 修复建议

## 编译选项

CUTE 内核编译使用以下选项：
```bash
nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
  -I{CUTLASS_ROOT}/include \
  -I/usr/local/cuda/include \
  -gencode arch=compute_80,code=sm_80 \
  -o kernel.so kernel.cu
```

关键点：
- `-std=c++17`: CUTE 需要 C++17
- `--shared -Xcompiler -fPIC`: 生成共享库
- `-gencode arch=compute_80,code=sm_80`: H100/H800 架构

## 手动测试

### 测试 Add 内核

```bash
cd /data/apps/project/cu2tri/cu2til/cases/triton2cute/add/add_simple
conda activate serve
python check_cute.py
```

### 仅编译不测试

```bash
python check_cute.py --compile-only
```

### 包含性能测试

```bash
python check_cute.py  # 默认包含性能测试
```

## 添加新测试用例

### 1. 创建目录结构

```bash
mkdir -p /data/apps/project/cu2tri/cu2til/cases/triton2cute/my_kernel/my_test/{triton_,cute_,torch_,logs}
```

### 2. 实现必要文件

- **get_data.py**: 实现 `Params` 类和 `get_triton_torch_inputs()` 函数
- **triton_/kernel.py**: Triton 源代码，实现 `triton_kernel()` 函数
- **torch_/ref.py**: PyTorch 参考，实现 `torch_kernel()` 函数
- **check_cute.py**: 从模板复制并根据需要修改

### 3. 更新配置

在 `config/case_config.yaml` 中添加：
```yaml
my_testset:
  mode: scan
  path: cu2til/cases/triton2cute/my_kernel
  include_ops: []
  exclude_ops: []
```

在 `config/args.py` 的 testset choices 中添加你的测试集名称。

## 注意事项

### GPU 选择

- 系统使用 `CUDA_VISIBLE_DEVICES` 环境变量控制 GPU
- Flash Attention benchmark 脚本可能硬编码了 GPU ID
- 修改测试脚本中的 GPU 设置以匹配你的环境

### CUTLASS 版本

- 系统针对 CUTLASS 3.x 设计
- 确保 CUTLASS_ROOT 指向正确版本
- 如果使用其他版本，可能需要调整 prompt

### 数据类型

- Add 示例使用 float32
- Flash Attention 示例使用 float16
- 根据需求在 `get_data.py` 中配置

### 编译超时

- CUTE 代码编译可能较慢（模板展开）
- 默认超时设置为 5 分钟
- 可在 `services/testing_cute.py` 中调整

## 故障排除

### 编译错误

1. **找不到 CUTE 头文件**
   ```bash
   export CUTLASS_ROOT=/path/to/cutlass
   ```

2. **架构不匹配**
   - 修改 `check_cute.py` 中的 `-gencode` 参数
   - 例如：SM86 用 `arch=compute_86,code=sm_86`

3. **C++ 标准错误**
   - 确保使用 `-std=c++17` 或更高

### 运行时错误

1. **加载 .so 失败**
   - 检查编译是否成功
   - 验证函数签名匹配

2. **结果不匹配**
   - 检查数据类型是否一致
   - 增加容差 (`rtol`, `atol`)
   - 检查 layout 和 stride 是否正确

### LLM 翻译问题

1. **生成的代码不完整**
   - 检查 prompt 是否清晰
   - 尝试不同的模型
   - 增加 max_rounds

2. **频繁失败**
   - 简化测试用例
   - 提供更多示例
   - 在 prompt 中添加特定约束

## 高级配置

### 使用自定义 Prompt

修改 `cu2til/prompt/triton2cute.py` 中的模板：
- `complex_initial_prompt`: 详细指导
- `simple_initial_prompt`: 简化指导
- `feedback_prompt`: 错误反馈

### 添加新的编译选项

修改 `check_cute.py` 中的 `compile_cute_kernel()` 函数：
```python
compile_cmd = [
    'nvcc',
    '-std=c++17',
    '-O3',
    # 添加你的选项
    '--use_fast_math',  # 例如
    ...
]
```

## 参考资料

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTE Tutorial](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Triton Documentation](https://triton-lang.org/)

## 示例输出

成功的测试运行输出示例：

```
Compiling CUTE kernel...
Command: nvcc -std=c++17 -O3 --shared ...
Compilation successful: cute_/kernel.so

=== Correctness Test ===
Max difference: 0.0001
Mean difference: 0.00003
PASSED: Results match!

All tests completed successfully!
```

## 联系和贡献

如有问题或建议，请提交 issue 或 PR。

