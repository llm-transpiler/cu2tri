# gpu-detect

自动检测系统GPU环境并生成NVGPU配置文件。

## 功能

- 自动检测系统中所有可用的GPU（使用nvidia-smi或pynvml）
- 生成完整的YAML配置文件
- 提供正确的nvgpu-server启动命令
- 支持自定义GPU模式、内存阈值、并发任务数等参数

## 快速使用

### 基本用法

```bash
# 自动检测GPU并生成配置文件
cd /cu2tri
python -m server.nvgpu.tools.gpu_detect

# 检测并打印启动命令
python -m server.nvgpu.tools.gpu_detect --print-command

# 预览配置（不写入文件）
python -m server.nvgpu.tools.gpu_detect --dry-run
```

### 自定义配置

```bash
# 指定GPU模式为独占
python -m server.nvgpu.tools.gpu_detect --mode exclusive

# 设置最大并发任务数
python -m server.nvgpu.tools.gpu_detect --max-tasks 10

# 设置内存阈值
python -m server.nvgpu.tools.gpu_detect --memory-threshold 0.8

# 组合使用
python -m server.nvgpu.tools.gpu_detect \
    --mode exclusive \
    --max-tasks 5 \
    --memory-threshold 0.85 \
    --print-command
```

### 输出控制

```bash
# 指定输出文件路径
python -m server.nvgpu.tools.gpu_detect --output /path/to/my_config.yml

# 仅输出到终端
python -m server.nvgpu.tools.gpu_detect --dry-run
```

### 检测方法

```bash
# 自动选择（优先pynvml，fallback到nvidia-smi）
python -m server.nvgpu.tools.gpu_detect --method auto

# 强制使用nvidia-smi
python -m server.nvgpu.tools.gpu_detect --method nvidia-smi

# 强制使用pynvml
python -m server.nvgpu.tools.gpu_detect --method pynvml
```

## 生成的配置文件

配置文件默认保存在 `server/nvgpu/configs/gpu_resources/` 目录下，文件名格式为 `{hostname}_gpu_config.yml`。

```yaml
# Auto-generated NVGPU GPU configuration
# Generated: 2026-03-08 12:34:56
# Total GPUs: 2

gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "NVIDIA RTX 6000 Ada Generation"
    uuid: "GPU-1b8c43a8-35ba-a01b-a2ee-0d6fe8f1f60f"
    memory_gb: 49.14
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3

  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    name: "NVIDIA RTX 6000 Ada Generation"
    uuid: "GPU-78b50b8e-ef33-b3fd-2c52-e3f135990f31"
    memory_gb: 49.14
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3

server:
  auto_register_gpus: true
```

## 启动服务器

生成配置后，使用打印的命令启动服务器：

```bash
cd /cu2tri
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/{hostname}_gpu_config.yml
```

或使用Python模块方式：

```bash
cd /cu2tri
python -m server.nvgpu.main --gpu-config server/nvgpu/configs/gpu_resources/{hostname}_gpu_config.yml
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output`, `-o` | 输出配置文件路径 | 自动生成（基于主机名） |
| `--mode`, `-m` | GPU模式 (shared/exclusive) | shared |
| `--memory-threshold` | 内存阈值 (0.0-1.0) | 0.75 |
| `--max-tasks` | 每GPU最大并发任务数 | 3 |
| `--method` | 检测方法 (auto/nvidia-smi/pynvml) | auto |
| `--dry-run` | 预览配置不写入文件 | - |
| `--print-command` | 打印启动命令 | - |
| `--config-dir` | 配置文件目录 | server/nvgpu/configs/gpu_resources/ |

## 完整示例

```bash
# 1. 检测GPU并生成配置
cd /cu2tri
python -m server.nvgpu.tools.gpu_detect \
    --mode exclusive \
    --max-tasks 5 \
    --memory-threshold 0.8 \
    --print-command

# 2. 使用生成的配置启动服务器
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/your_hostname_gpu_config.yml

# 3. 验证服务器运行
curl http://localhost:8080/health
curl http://localhost:8080/gpus
```

## 故障排查

### nvidia-smi not found

确保NVIDIA驱动已正确安装：

```bash
nvidia-smi
```

### pynvml not available

安装pynvml库：

```bash
pip install nvidia-ml-py3
```

### 配置文件路径问题

始终从项目根目录 `/cu2tri` 运行命令，配置文件路径是相对于根目录的。
