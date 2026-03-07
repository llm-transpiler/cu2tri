# cu2tri - CUDA 到 Triton 转译器与 GPU 任务调度系统

## 概述

cu2tri 是一个模块化平台，提供：
- **GPU 任务调度** - 多 GPU 任务管理与 REST API
- **LLM 驱动转译** - CUDA 到 Triton 代码自动转译
- **性能测试** - 自动化内核评估与基准测试

## 项目结构

```
/cu2tri/
├── server/                   # 设备服务器
│   ├── nvgpu/              # NVIDIA GPU 服务器
│   │   ├── main.py         # 服务器入口
│   │   ├── api_server.py   # FastAPI REST API
│   │   ├── gpu_manager.py  # GPU 管理
│   │   ├── scheduler.py    # 任务调度
│   │   ├── task_queue.py   # 任务队列
│   │   ├── task_runner.py  # 任务执行
│   │   ├── client.py       # Python 客户端
│   │   └── configs/        # GPU 配置文件
│   │       └── gpu_resources/
│   │           ├── P250_A6000.yml
│   │           └── g0015_H800.yml
│   └── common/            # 服务器共享代码
│       ├── timezone.py    # 时区工具
│       ├── task_refs.py   # 任务引用格式化
│       ├── context.py     # 上下文管理
│       ├── timer.py       # 计时工具
│       └── logger.py      # 日志工具
│
├── llm_trans/              # LLM 转译器 (项目级)
│   ├── cli.py             # 命令行接口
│   ├── clients/           # LLM 客户端
│   ├── config/            # 配置文件
│   ├── core/              # 核心运行时
│   ├── data/              # 数据模型
│   ├── io/                # 输入输出
│   ├── services/          # 业务逻辑
│   ├── utils/             # 工具函数
│   ├── prompts/           # 提示词模板
│   ├── cases/             # 测试用例
│   └── runs/              # 运行结果
│
├── llm/                    # LLM 提供商接口
│   ├── client/            # 客户端实现
│   ├── providers/         # 提供商实现
│   └── history/           # 对话历史
│
├── scripts/                # 实用脚本
├── docs/                   # 文档
│   ├── 架构设计.md
│   ├── nvgpu服务器指南.md
│   └── llm转译器指南.md
├── pyproject.toml         # 项目配置
└── README.md
```

## 快速开始

### 1. 安装

```bash
cd /cu2tri
pip install -e .
```

### 2. 启动 NVGPU 服务器

```bash
# 设置环境变量
export PROJECT_ROOT=/cu2tri

# 启动服务器
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

服务器将在 `http://0.0.0.0:8080` 启动。

**自定义端口：**
```bash
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml --port 8848
```

### 3. 运行 LLM 转译器

```bash
# 设置环境变量
export PROJECT_ROOT=/cu2tri

# 基本转译
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler --first-only

# 使用 NVGPU 服务器
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler --use-nvgpu

# 高并发运行
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler --concurrency 30 --temperature 1.0 --max-attempts 1
```

## 环境变量

```bash
# 项目根目录 (必需)
export PROJECT_ROOT=/cu2tri

# LLM API 密钥 (按需设置)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...
```

## 配置文件

### GPU 配置

编辑 `server/nvgpu/configs/gpu_resources/*.yml` 配置：
- 可用 GPU
- 内存阈值
- 并发任务限制

### 模型配置

编辑 `llm_trans/config/model_clients.yaml` 配置：
- LLM 提供商和端点
- 模型别名
- API 密钥和基础 URL

## 启动方式

### NVGPU 服务器

```bash
# 方式 1: 使用安装的命令 (推荐)
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 方式 2: 使用 Python 模块
python -m server.nvgpu.main --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

### LLM 转译器

```bash
# 方式 1: 使用安装的命令 (推荐)
llm-trans --model gpt_oss_120b_local_5880x4 --testset xpiler

# 方式 2: 使用 Python 模块
python -m llm_trans --model gpt_oss_120b_local_5880x4 --testset xpiler
```

## CLI 参数

### nvgpu-server

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu-config` | GPU 配置文件路径 | 必需 |
| `--host` | 服务器地址 | `0.0.0.0` |
| `--port` | 服务器端口 | `8080` |
| `--log-level` | 日志级别 | `INFO` |

### llm-trans

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 | 必需 |
| `--testset` | 测试集名称 | `xpiler` |
| `--direction` | 转译方向 | `cu2tri` |
| `--first-only` | 只运行第一个用例 | `false` |
| `--case-types` | 用例类型过滤 | 全部 |
| `--max-rounds` | 最大对话轮数 | `3` |
| `--temperature` | 生成温度 | `0.7` |
| `--max-attempts` | 最大重试次数 | `5` |
| `--concurrency` | 并发任务数 | `10` |
| `--use-nvgpu` | 使用 NVGPU 服务器 | `false` |

## 文档

- [架构设计.md](docs/架构设计.md) - 系统架构说明
- [nvgpu服务器指南.md](docs/nvgpu服务器指南.md) - NVGPU 服务器详细指南
- [llm转译器指南.md](docs/llm转译器指南.md) - LLM 转译器详细指南

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 格式化代码
black .
```

## 许可证

[请指定您的许可证]
