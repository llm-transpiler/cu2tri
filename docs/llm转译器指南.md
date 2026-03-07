# LLM 转译器完整指南

## 概述

LLM 转译器 (`llm_trans`) 是一个基于大语言模型的代码转译工具，主要用于 CUDA 到 Triton 的自动转换。它支持多种 LLM 提供商，自动处理转译任务，并支持重试和性能测试。

## 功能特性

- **多 LLM 支持**: OpenAI、Anthropic、本地模型、NVGPU 服务器
- **自动转译**: CUDA → Triton，Triton → CUTE
- **重试机制**: 失败自动重试，支持多次尝试
- **性能测试**: 转译后代码的正确性和性能验证
- **测试用例管理**: 290+ xpiler 测试用例
- **结果保存**: 完整的转译历史和结果存储
- **高并发**: 支持大规模并行转译

## 安装

```bash
cd /cu2tri
pip install -e .
```

## 快速开始

### 1. 设置环境变量

```bash
export PROJECT_ROOT=/cu2tri
```

### 2. 基本运行

```bash
# 快速测试 (单个用例)
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --first-only \
  --max-attempts 1 \
  --max-rounds 1

# 完整测试 (所有用例)
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler

# 高并发测试
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --concurrency 30 \
  --temperature 1.0 \
  --max-attempts 1
```

## 配置文件

### 1. 模型配置 (`config/model_clients.yaml`)

```yaml
models:
  # OpenAI 官方 API
  openai:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4"

  # 本地模型服务器 (如 vLLM)
  gpt_oss_120b_local_5880x4:
    type: "openai"
    base_url: "http://192.168.1.100:8848/v1"
    api_key: "none"
    model: "gpt-oss-120b"

  # 使用 NVGPU 服务器
  nvgpu:
    type: "nvgpu"
    server_url: "http://localhost:8080"
    gpu_requirements:
      gpu_count: 1
      memory_per_gpu: 20000000000
    timeout: 3600

  # Anthropic Claude
  claude:
    type: "anthropic"
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-4-sonnet"
```

### 2. 测试用例配置 (`config/case_config.yaml`)

```yaml
testsets:
  xpiler:
    path: "cases/xpiler"
    case_types:
      - elementwise    # 逐元素操作
      - reduction      # 归约操作
      - mha            # 多头注意力
      - gqa            # 分组查询注意力
      - convolution    # 卷积操作

output:
  base_dir: "runs"
  save_failed: true
  save_successful: true
```

### 3. 转译方向配置 (`config/directions.yaml`)

```yaml
directions:
  cu2tri:    # CUDA 到 Triton
    name: "CUDA to Triton"
    prompt_template: "prompts/cuda2triton.py"
    input_ext: ".cu"
    output_ext: ".triton.py"

  tri2cute:  # Triton 到 CUTE
    name: "Triton to CUTE"
    prompt_template: "prompts/triton2cute.py"
    input_ext: ".triton.py"
    output_ext: ".cute.hpp"
```

## 启动方式

### 方式 1: 使用安装的命令 (推荐)

```bash
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler
```

### 方式 2: 使用 Python 模块

```bash
python -m llm_trans --model gpt_oss_120b_local_5880x4 --testset xpiler
```

> **注意**: 由于 Python 模块使用相对导入，不支持直接运行 `cli.py` 文件。请使用上述两种方式之一。

## 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型名称 (在 model_clients.yaml 中定义) | `gpt_oss_120b_local_5880x4` |

### 可选参数

| 参数 | 说明 | 默认值 | 常用值 |
|------|------|--------|--------|
| `--testset` | 测试集名称 | `xpiler` | `xpiler`, `xpiler_extended` |
| `--direction` | 转译方向 | `cu2tri` | `cu2tri`, `tri2cute` |
| `--first-only` | 只运行每个类型的第一个用例 | `false` | - |
| `--case-types` | 指定用例类型 | 全部 | `elementwise,reduction` |
| `--skip-case-types` | 跳过指定类型 | 无 | `convolution` |
| `--max-rounds` | 最大对话轮数 | `3` | `1`, `3`, `5` |
| `--temperature` | 生成温度 | `0.7` | `0.5`, `1.0` |
| `--max-attempts` | 最大重试次数 | `5` | `1`, `3`, `5` |
| `--concurrency` | 并发任务数 | `10` | `5`, `30` |
| `--use-nvgpu` | 使用 NVGPU 服务器 | `false` | - |
| `--nvgpu-server` | NVGPU 服务器地址 | `localhost:8080` | - |
| `--outputs-root` | 输出目录 | `runs` | - |
| `--attempt-policy` | 尝试策略 | `first_success` | `first_success`, `exhaustive` |

## 常用命令示例

### 快速测试

```bash
# 最快测试 (单用例，不重试)
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --first-only \
  --max-attempts 1 \
  --max-rounds 1
```

### 类型过滤

```bash
# 只测试 elementwise 类型
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --case-types elementwise

# 跳过卷积类用例
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --skip-case-types convolution
```

### 高性能运行

```bash
# 高并发 + 不重试
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --concurrency 30 \
  --max-attempts 1 \
  --temperature 1.0
```

### 使用 NVGPU 服务器

```bash
# 先启动 NVGPU 服务器
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml &

# 使用 NVGPU 运行
llm-trans \
  --model nvgpu \
  --testset xpiler \
  --nvgpu-server http://localhost:8080
```

### 调试模式

```bash
# 单个用例，详细日志
llm-trans \
  --model gpt_oss_120b_local_5880x4 \
  --testset xpiler \
  --first-only \
  --max-rounds 5 \
  --temperature 0.5
```

## 输出结构

运行结果保存在 `runs/_cu2tri_runs/` 目录：

```
runs/_cu2tri_runs/
└── xpiler/                          # 测试集
    └── gpt_oss_120b_local_5880x4/   # 模型
        └── 20250308_120000/         # 时间戳
            └── add_1024_1024_1024/  # 用例名称
                └── attempt_1/       # 尝试次数
                    ├── cuda_        # CUDA 输入
                    │   └── kernel.cu
                    ├── triton_      # Triton 输出
                    │   └── kernel.py
                    ├── torch_       # 参考实现
                    │   └── ref.py
                    ├── check_triton.py
                    ├── check_cuda.py
                    └── result.json  # 详细结果
```

### 结果文件说明

**result.json**:
```json
{
  "case_name": "add_1024_1024_1024",
  "attempt_number": 1,
  "round_number": 1,
  "status": "success",           // success, failed, error
  "translation_successful": true,
  "compilation_successful": true,
  "test_passed": true,
  "error_message": null,
  "timing": {
    "translation_time_ms": 1234,
    "compilation_time_ms": 567,
    "test_time_ms": 890
  }
}
```

## Python API

```python
from llm_trans.cli import prepare_context
from llm_trans.services.runner import run
import asyncio

# 准备运行上下文
context = prepare_context([
    "--model", "gpt_oss_120b_local_5880x4",
    "--testset", "xpiler",
    "--first-only"
])

# 运行转译
asyncio.run(run(context))
```

## 测试用例

### 查看可用测试用例

```bash
# 列出所有测试用例
find llm_trans/cases/xpiler -name "*.cu" | head -20

# 按类型统计
find llm_trans/cases/xpiler -type d -name "add_*" | wc -l
```

### 用例类型

| 类型 | 说明 | 数量 |
|------|------|------|
| `elementwise` | 逐元素操作 (add, mul, sub) | ~50 |
| `reduction` | 归约操作 (sum, max, min) | ~40 |
| `mha` | 多头注意力 | ~30 |
| `gqa` | 分组查询注意力 | ~30 |
| `convolution` | 卷积操作 | ~20 |

### 添加新测试用例

1. 创建用例目录：
```bash
mkdir -p llm_trans/cases/xpiler/elementwise/my_new_kernel
```

2. 添加输入文件：
```bash
# CUDA 输入 (必需)
cat > llm_trans/cases/xpiler/elementwise/my_new_kernel/input.cu << 'EOF'
__global__ void my_kernel(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * 2.0f;
    }
}
EOF
```

3. 添加参考实现 (可选)：
```bash
cat > llm_trans/cases/xpiler/elementwise/my_new_kernel/reference.py << 'EOF'
import torch

def my_kernel(x):
    return x * 2.0
EOF
```

## 故障排除

### 1. 模块导入错误

**错误**: `ModuleNotFoundError: No module named 'llm_trans'`

**解决**:
```bash
# 重新安装
cd /cu2tri
pip install -e .

# 或设置 PYTHONPATH
export PYTHONPATH=/cu2tri:$PYTHONPATH
```

### 2. 模型连接失败

**错误**: `Connection refused` 或 `Timeout`

**解决**:
```bash
# 检查模型服务是否运行
curl http://192.168.1.100:8848/v1/models

# 检查配置
cat llm_trans/config/model_clients.yaml

# 测试连接
timeout 5 curl http://192.168.1.100:8848/v1/chat/completions
```

### 3. 并发过高失败

**错误**: 大量请求超时或失败

**解决**:
```bash
# 降低并发数
llm-trans --model xxx --testset xpiler --concurrency 5

# 增加超时时间
llm-trans --model xxx --testset xpiler --max-attempts 3
```

### 4. GPU 内存不足

**错误**: `CUDA out of memory`

**解决**:
```bash
# 减少并发数
llm-trans --model xxx --testset xpiler --concurrency 3

# 或使用 NVGPU 服务器进行任务调度
nvgpu-server --gpu-config ... &
llm-trans --model nvgpu --testset xpiler
```

### 5. 配置文件未找到

**错误**: `FileNotFoundError: config/model_clients.yaml`

**解决**:
```bash
# 设置项目根目录
export PROJECT_ROOT=/cu2tri

# 检查配置文件
ls llm_trans/config/
```

## 性能优化

### 1. 并发调优

```bash
# 稳定性优先 (低并发)
llm-trans --model xxx --testset xpiler --concurrency 5

# 速度优先 (高并发)
llm-trans --model xxx --testset xpiler --concurrency 30

# 平衡
llm-trans --model xxx --testset xpiler --concurrency 10
```

### 2. 减少重试

```bash
# 快速测试 (不重试)
llm-trans --model xxx --testset xpiler --max-attempts 1 --max-rounds 1

# 完整测试 (多次重试)
llm-trans --model xxx --testset xpiler --max-attempts 5 --max-rounds 3
```

### 3. 温度调整

```bash
# 保守 (更稳定，但可能缺乏创意)
llm-trans --model xxx --testset xpiler --temperature 0.3

# 平衡
llm-trans --model xxx --testset xpiler --temperature 0.7

# 激进 (更多创意，但可能不稳定)
llm-trans --model xxx --testset xpiler --temperature 1.0
```

### 4. 类型过滤

```bash
# 只测试特定类型
llm-trans --model xxx --testset xpiler --case-types elementwise,reduction

# 跳过复杂类型
llm-trans --model xxx --testset xpiler --skip-case-types convolution,mha,gqa
```

## 日志和调试

### 查看日志

```bash
# 日志输出到控制台和文件
llm-trans --model xxx --testset xpiler 2>&1 | tee trans.log

# 只看错误
llm-trans --model xxx --testset xpiler 2>&1 | grep ERROR
```

### 调试单个用例

```bash
# 查看用例详情
cat llm_trans/cases/xpiler/elementwise/add_1024_1024_1024/input.cu

# 手动测试
llm-trans \
  --model xxx \
  --testset xpiler \
  --case-types elementwise \
  --first-only \
  --max-rounds 5
```

## 最佳实践

### 1. 开发阶段

```bash
# 快速迭代
llm-trans \
  --model xxx \
  --testset xpiler \
  --first-only \
  --max-attempts 1 \
  --max-rounds 1 \
  --temperature 0.5
```

### 2. 测试阶段

```bash
# 完整测试
llm-trans \
  --model xxx \
  --testset xpiler \
  --concurrency 10 \
  --max-attempts 3 \
  --temperature 0.7
```

### 3. 生产阶段

```bash
# 大规模运行
llm-trans \
  --model xxx \
  --testset xpiler \
  --concurrency 30 \
  --max-attempts 1 \
  --temperature 1.0
```
