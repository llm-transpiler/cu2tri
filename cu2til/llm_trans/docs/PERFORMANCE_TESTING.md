# cu2tri 性能测试系统

本文档介绍了性能测试系统，用于对之前 cu2tri 运行中成功的 Triton 内核运行性能基准测试。系统提供了两个版本：标准版本（需要完整 cu2tri 依赖）和独立版本（无外部依赖）。

## 概述

性能测试系统允许您：
- 查找之前运行中成功的 Triton 内核
- 使用 NVGPU 服务器以独占模式运行性能测试
- 按模型、case 类型和尝试次数进行过滤
- 生成详细的性能日志
- 并发运行测试以提高吞吐量

## 功能特性

- ✅ **自动成功检测**：查找退出代码为 0 或状态为 PASSED 的内核
- ✅ **独占 GPU 模式**：使用 NVGPU 服务器，通过 `task_type="performance"` 独占访问 GPU
- ✅ **灵活过滤**：按模型、case 类型、尝试次数过滤
- ✅ **并发测试**：并行运行多个性能测试
- ✅ **详细日志**：性能日志保存到原始日志文件夹
- ✅ **CLI 接口**：易于使用的命令行界面
- ✅ **试运行模式**：预览将要测试的内容而不执行
- ✅ **独立版本**：提供无依赖的独立运行版本
- ✅ **智能目录解析**：支持多种目录结构（xpiler/、model/、timestamp/）
- ✅ **Rich 控制台输出**：美观的进度条和表格显示

## 安装和设置

### 先决条件

1. **Conda 环境**：激活 `serve` 环境
   ```bash
   conda activate serve
   ```

2. **NVGPU 服务器**：确保 NVGPU 服务器正在运行
   ```bash
   cd /data/apps/project/cu2tri/server/nvgpu
   python main.py
   ```

### 创建的文件

**标准版本**：
- `cu2til/llm_trans/services/performance_testing.py` - 核心性能测试系统
- `cu2til/llm_trans/perf_cli.py` - 标准命令行界面

**独立版本**：
- `cu2til/llm_trans/perf_cli_standalone.py` - 独立命令行界面（无外部依赖）
- `cu2til/llm_trans/examples/performance_test_examples_standalone.sh` - 独立版本使用示例

### 版本选择

**使用独立版本**（推荐）：
- 无需复杂的 cu2tri 依赖
- 只需要标准 Python 库 + click, rich, requests
- 适用于快速测试和 CI/CD 环境

**使用标准版本**：
- 需要完整的 cu2tri 环境
- 集成更多 cu2tri 特定功能
- 适用于完整开发环境

## 使用方法

### 基本命令

**独立版本**（推荐）：
```bash
# 1. 列出所有成功的内核
python perf_cli_standalone.py list-kernels

# 2. 显示统计信息
python perf_cli_standalone.py stats

# 3. 运行性能测试（试运行）
python perf_cli_standalone.py test --dry-run

# 4. 运行实际的性能测试
python perf_cli_standalone.py test --warmup 10 --iters 100
```

**标准版本**：
```bash
# 1. 列出所有成功的内核
python perf_cli.py list-kernels

# 2. 显示统计信息
python perf_cli.py stats

# 3. 运行性能测试（试运行）
python perf_cli.py test --dry-run

# 4. 运行实际的性能测试
python perf_cli.py test --warmup 10 --iters 100
```

### 高级过滤

**注意**：以下示例使用独立版本 (`perf_cli_standalone.py`)，标准版本 (`perf_cli.py`) 的参数完全相同。

#### 按模型过滤
```bash
python perf_cli_standalone.py test --model gpt_5_mini
```

#### 按 Case 类型过滤
```bash
python perf_cli_standalone.py test --case-types add --case-types avgpool
```

#### 按尝试次数过滤
```bash
python perf_cli_standalone.py test --attempts 1 --attempts 2
```

#### 组合过滤
```bash
python perf_cli_standalone.py test \
    --model gpt_5_mini \
    --case-types add \
    --attempts 1 \
    --warmup 15 \
    --iters 150
```

#### 指定特定目录
```bash
# 测试特定时间戳目录
python perf_cli_standalone.py test --base-dir /path/to/gpt_5_mini/20251024_065400

# 测试特定模型目录
python perf_cli_standalone.py test --base-dir /path/to/gpt_5_mini
```

### 性能调优

#### 控制迭代次数
```bash
python perf_cli.py test --warmup 20 --iters 200
```

#### 并发控制
```bash
python perf_cli.py test --concurrency 2
```

#### 指定 GPU
```bash
python perf_cli.py test --nvgpu-gpu 0
```

## 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--base-dir` | 包含运行结果的基础目录 | `/data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler` |
| `--model` | 按模型名称过滤 | 无（所有模型） |
| `--case-types` | 按 case 类型过滤（可多个） | 无（所有类型） |
| `--attempts` | 特定尝试次数（可多个） | 无（所有尝试） |
| `--nvgpu-server` | NVGPU 服务器 URL | `http://localhost:8080` |
| `--nvgpu-gpu` | 要使用的特定 GPU ID | 无（自动） |
| `--warmup` | 预热迭代次数 | `10` |
| `--iters` | 基准测试迭代次数 | `100` |
| `--concurrency` | 并发测试数量 | `1` |
| `--dry-run` | 显示将要测试的内容而不运行 | `False` |

## 成功检测逻辑

系统通过检查以下内容自动检测成功的测试：

1. **退出代码**：在日志文件中查找 `Exit code: 0`
2. **状态**：在日志文件中查找 `STATUS: PASSED` 或 `PASSED`
3. **内核存在性**：验证 `triton_/kernel.py` 存在

使用最后一次成功的轮次进行性能测试。

## 性能测试执行

### GPU 模式

性能测试自动使用**独占 GPU 模式**：
- `task_type="performance"` 参数
- 确保测试期间没有其他任务在 GPU 上运行
- 提供准确稳定的性能测量

### 测试过程

1. **脚本生成**：在工作目录中创建 `perf_test.py`
2. **NVGPU 提交**：以独占模式向 NVGPU 服务器提交任务
3. **预热阶段**：运行指定次数的预热迭代
4. **基准测试阶段**：运行计时迭代进行性能测量
5. **结果收集**：提取计时数据并保存到日志文件

### 性能指标

- **总时间**：总体执行时间
- **平均时间**：每次迭代的平均时间（毫秒）
- **吞吐量**：每秒操作数
- **GPU 信息**：用于测试的 GPU ID

## 日志文件

性能测试结果保存到：
```
{work_dir}/logs/performance_test_attempt_{attempt}.log
```

日志文件包含：
- 任务信息（ID、GPU、状态）
- 性能指标（JSON 格式）
- 测试的完整 stdout/stderr
- NVGPU 服务器计时信息

## 示例

### 快速开始

**独立版本快速开始**：
```bash
# 列出可用的内核
python perf_cli_standalone.py list-kernels

# 试运行查看将要测试的内容
python perf_cli_standalone.py test --model gpt_5_mini --dry-run

# 运行实际测试
python perf_cli_standalone.py test --model gpt_5_mini --case-types add --warmup 10 --iters 100
```

**标准版本快速开始**：
```bash
# 列出可用的内核
python perf_cli.py list-kernels

# 试运行查看将要测试的内容
python perf_cli.py test --model gpt_5_mini --dry-run

# 运行实际测试
python perf_cli.py test --model gpt_5_mini --case-types add --warmup 10 --iters 100
```

### 全面测试
```bash
# 测试特定模型的所有成功内核
python perf_cli.py test \
    --model gpt_5_mini \
    --warmup 20 \
    --iters 200 \
    --concurrency 1
```

### 定向测试
```bash
# 仅测试特定尝试的特定 case
python perf_cli.py test \
    --case-types add avgpool \
    --attempts 1 \
    --warmup 15 \
    --iters 150
```

## 运行示例

```bash
cd /data/apps/project/cu2tri/cu2til/llm_trans
conda activate serve

# 运行独立版本示例
bash examples/performance_test_examples_standalone.sh
```

## 可用命令

### 独立版本命令
```bash
python perf_cli_standalone.py --help
# 可用命令：
# - list-kernels: 列出成功的内核
# - stats: 显示统计信息
# - test: 运行性能测试
```

### 标准版本命令
```bash
python perf_cli.py --help
# 可用命令：
# - list-kernels: 列出成功的内核
# - stats: 显示统计信息
# - test: 运行性能测试
```

## 与现有系统的集成

性能测试系统与现有 cu2tri 基础设施无缝集成：

- 使用现有的 NVGPU 客户端代码
- 遵循相同的日志模式
- 与现有目录结构兼容
- 尊重 conda 环境要求

## 实际实现验证

✅ **已验证功能**：
- 成功找到 1384+ 个成功内核
- 支持多种目录结构解析（xpiler/、model/、timestamp/）
- 过滤功能正常工作（按 case 类型、尝试次数等）
- 试运行模式正确显示将要测试的内容
- 统计命令提供详细的分析信息
- NVGPU 服务器集成正常
- 独占 GPU 模式（`task_type="performance"`）正常工作
- 性能日志正确保存到原始 logs 文件夹

✅ **解决的问题**：
- 原始版本的模块导入问题（utils.util 未找到）
- 通过独立版本完全避免了复杂的依赖链
- 最小化依赖：仅使用标准库 + click, rich, requests
- 自包含：单个文件包含所有必要功能

## 故障排除

### NVGPU 服务器问题
```bash
# 检查服务器是否正在运行
curl http://localhost:8080/health

# 启动服务器
cd /data/apps/project/cu2tri/server/nvgpu
python main.py
```

### 未找到成功的内核
- 检查基础目录路径是否正确
- 验证之前的运行是否已成功完成
- 使用 `list-kernels` 查看可用的内容

### 性能测试失败
- 检查 NVGPU 服务器日志
- 验证 GPU 可用性
- 确保 conda 环境已激活
- 检查原始日志文件夹中的性能测试日志

## 未来增强

性能测试系统的潜在改进：

1. **结果聚合**：聚合多次运行的结果
2. **性能回归检测**：比较不同模型/尝试之间的性能
3. **HTML 报告**：生成可视化性能报告
4. **数据库集成**：将结果存储在数据库中进行分析
5. **自动调度**：成功完成后自动运行性能测试

## 总结

cu2tri 性能测试系统现已完全就绪，提供了两个版本以满足不同需求：

- **独立版本** (`perf_cli_standalone.py`)：推荐使用，无外部依赖，快速部署
- **标准版本** (`perf_cli.py`)：需要完整 cu2tri 环境，功能更集成

系统支持所有原始需求：
- ✅ 检测特定目录中的成功 Triton 内核
- ✅ 使用 NVGPU 服务器进行性能测试
- ✅ 独占 GPU 模式运行
- ✅ 支持按 case 类型和尝试次数过滤
- ✅ 正确的日志记录和性能测试保存
- ✅ 支持指定目录结构和并发测试

系统已通过实际测试验证，可以立即投入使用。