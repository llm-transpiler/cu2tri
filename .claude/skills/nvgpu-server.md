# NVGPU 服务器使用指南

## 启动服务器

### 方式 1: 使用安装的命令 (推荐)

```bash
# 指定 GPU 配置文件
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 或直接指定 GPU ID (简单场景)
nvgpu-server --gpus 0 1

# 指定端口
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml --port 8080
```

### 方式 2: 使用 Python 模块

```bash
python -m server.nvgpu.main --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

## 常见问题

### 问题: 任务一直 pending，没有 GPU 被注册

**原因**: GPU 配置文件中没有启用对应的 GPU，或没有指定正确的配置文件。

**解决**:
```bash
# 检查可用 GPU
nvidia-smi --list-gpus

# 方法1: 使用正确的配置文件
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 方法2: 直接指定 GPU ID
nvgpu-server --gpus 0 1

# 方法3: 创建自己的 GPU 配置文件
# 复制现有配置并修改 enabled: true 和 UUID
```

### 问题: 没有注册任何 GPU

检查日志:
```bash
tail -20 server/nvgpu/logs/nvgpu_server_*.log
```

如果看到 `Registered GPUs: []`，说明没有 GPU 被注册。使用 `--gpus` 参数强制注册。

### GPU 配置文件格式

```yaml
# server/nvgpu/configs/gpu_resources/your_config.yml
gpus:
  - logical_id: 0        # 系统逻辑 ID
    nvidia_smi_id: 0     # nvidia-smi 显示的 ID
    cuda_visible_id: 0   # CUDA_VISIBLE_DEVICES 映射
    name: "GPU 名称"
    uuid: "GPU-xxx"      # nvidia-smi -L 获取的 UUID
    enabled: true        # 启用此 GPU
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 10

server:
  auto_register_gpus: true
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu-config` | GPU 配置文件路径 | 必须指定 |
| `--gpus` | 直接指定 GPU ID | 无 |
| `--host` | 服务器地址 | 0.0.0.0 |
| `--port` | 服务器端口 | 8080 |
| `--gpu-mode` | GPU 模式 (exclusive/shared) | shared |
| `--lb-strategy` | 负载均衡策略 | round_robin |

## 获取 GPU UUID

```bash
nvidia-smi -L
# 输出示例:
# GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-1b8c43a8-35ba-a01b-a2ee-0d6fe8f1f60f)
# GPU 1: NVIDIA RTX 6000 Ada Generation (UUID: GPU-78b50b8e-ef33-b3fd-2c52-e3f135990f31)
```

## 检查服务器状态

```bash
# 健康检查
curl http://localhost:8080/health

# 查看队列状态
curl http://localhost:8080/stats

# 查看特定任务
curl http://localhost:8080/tasks/<task_id>
```
