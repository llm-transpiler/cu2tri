# NVGPU 服务器指南

## 概述

NVGPU 服务器是一个基于 FastAPI 的 GPU 任务调度系统，支持多 GPU 资源管理和任务并发执行。

## 功能特性

- **GPU 资源管理**: 支持多 GPU 卡的资源分配和监控
- **任务调度**: 基于优先级的任务队列调度
- **REST API**: 提供完整的 HTTP API 接口
- **任务执行**: 异步任务执行，支持超时和取消
- **状态监控**: 实时任务状态和 GPU 使用情况查询
- **负载均衡**: 支持多种负载均衡策略

## 安装

```bash
cd /cu2tri
pip install -e ".[nvgpu]"
```

## GPU 配置

### 查询 GPU 信息

```bash
# 查看 GPU 基本信息
nvidia-smi --query-gpu=name,memory.total,pci.bus_id,compute_cap --format=csv

# 查看详细 GPU 信息
nvidia-smi
```

### 创建 GPU 配置文件

GPU 配置文件位于 `server/nvgpu/configs/gpu_resources/`，示例：

```yaml
# P250_A6000.yml
gpus:
  - name: "NVIDIA RTX 6000 Ada Generation"
    logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    uuid: null
    memory_gb: null
    enabled: true
    default_mode: "shared"           # exclusive 或 shared
    memory_threshold: 0.75           # 内存使用阈值 (0-1)
    max_concurrent_tasks: 10         # 最大并发任务数

  - name: "NVIDIA RTX 6000 Ada Generation"
    logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    uuid: null
    memory_gb: null
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 10

server:
  host: "0.0.0.0"
  port: 8080
  lb_strategy: "round_robin"         # 负载均衡策略
```

创建自定义配置：

```bash
# 复制模板
cp server/nvgpu/configs/gpu_resources/P250_A6000.yml \
   server/nvgpu/configs/gpu_resources/MY_GPU.yml

# 编辑配置
nano server/nvgpu/configs/gpu_resources/MY_GPU.yml
```

### 负载均衡策略

| 策略 | 说明 |
|------|------|
| `round_robin` | 轮询分配任务 |
| `least_loaded` | 分配给负载最轻的 GPU |
| `fill` | 优先填满第一个 GPU |

## 启动服务器

### 方式 1: 使用安装的命令 (推荐)

```bash
# 基础启动
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 自定义端口
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml --port 8848

# 指定日志级别
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml --log-level DEBUG
```

### 方式 2: 使用 Python 模块

```bash
# 从项目根目录
cd /cu2tri
python -m server.nvgpu.main --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 从任何目录 (需设置 PYTHONPATH)
export PYTHONPATH=/cu2tri:$PYTHONPATH
python -m server.nvgpu.main --gpu-config /cu2tri/server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

### 方式 3: 直接运行文件

```bash
cd /cu2tri
python server/nvgpu/main.py --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

### 环境变量

```bash
# 项目根目录 (推荐设置)
export PROJECT_ROOT=/cu2tri

# 指定 GPU 配置文件
export NVGPU_CONFIG=server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

## API 接口

服务器启动后访问 `http://localhost:8080` 查看自动生成的 API 文档。

### 1. 健康检查

```bash
curl http://localhost:8080/api/v1/health
```

### 2. 提交任务

```bash
POST /api/v1/tasks
Content-Type: application/json

{
  "task_id": "task-001",
  "command": ["python", "-m", "my_script"],
  "workdir": "/path/to/workdir",
  "env": {"CUDA_VISIBLE_DEVICES": "0"},
  "gpu_requirements": {
    "gpu_count": 1,
    "memory_per_gpu": 10000000000
  },
  "timeout": 3600
}
```

### 3. 查询任务状态

```bash
GET /api/v1/tasks/{task_id}

Response:
{
  "task_id": "task-001",
  "status": "running",
  "gpu_id": "gpu-0",
  "start_time": 1234567890,
  "result": null
}
```

### 4. 取消任务

```bash
POST /api/v1/tasks/{task_id}/cancel
```

### 5. 获取 GPU 状态

```bash
GET /api/v1/gpus

Response:
{
  "gpus": [
    {
      "logical_id": 0,
      "status": "busy",
      "memory_used": 10000000000,
      "task_id": "task-001"
    }
  ]
}
```

## Python 客户端

```python
from server.nvgpu.client import NVGPUClient

# 创建客户端
client = NVGPUClient("http://localhost:8080")

# 健康检查
is_healthy = client.health_check()
print(f"Server healthy: {is_healthy}")

# 提交任务
task_id = await client.submit_task(
    task_id="my-task",
    command=["python", "train.py"],
    workdir="/path/to/project",
    gpu_count=1
)

# 查询状态
status = await client.get_task_status(task_id)
print(status)

# 等待完成
result = await client.wait_for_task(task_id, timeout=3600)

# 取消任务
await client.cancel_task(task_id)
```

## 与 LLM 转译器集成

### 1. 配置 NVGPU 客户端

编辑 `llm_trans/config/model_clients.yaml`：

```yaml
nvgpu:
  type: "nvgpu"
  server_url: "http://localhost:8080"
  gpu_requirements:
    gpu_count: 1
    memory_per_gpu: 20000000000
  timeout: 3600
```

### 2. 启动 NVGPU 服务器

```bash
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
```

### 3. 运行转译器

```bash
llm-trans --model nvgpu --testset xpiler
```

## 日志

### 日志位置

```
server/nvgpu/logs/
├── nvgpu_server_20260308_052300.log    # 当前会话日志
└── nvgpu_server.log                     # 历史日志 (追加)
```

### 日志级别

| 级别 | 说明 |
|------|------|
| `DEBUG` | 详细调试信息 |
| `INFO` | 一般信息 (默认) |
| `WARNING` | 警告信息 |
| `ERROR` | 错误信息 |

### 查看日志

```bash
# 实时查看当前日志
tail -f server/nvgpu/logs/nvgpu_server_$(date +%Y%m%d)*.log

# 查看历史日志
tail -f server/nvgpu/logs/nvgpu_server.log

# 搜索错误
grep ERROR server/nvgpu/logs/nvgpu_server.log
```

## 故障排除

### 1. GPU 不可用

**问题**: `pynvml not available, GPU monitoring disabled`

**解决**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 安装 pynvml
pip install pynvml

# 检查 GPU 设备
ls /dev/nvidia*
```

### 2. 端口被占用

**问题**: `Address already in use`

**解决**:
```bash
# 查找占用进程
lsof -i :8080

# 杀死占用进程
kill -9 <PID>

# 或使用其他端口
nvgpu-server --gpu-config ... --port 8848
```

### 3. 权限问题

**问题**: `Permission denied` 访问 GPU

**解决**:
```bash
# 添加用户到 video 组
sudo usermod -a -G video $USER

# 重新登录生效
```

### 4. 导入错误

**问题**: `ModuleNotFoundError: No module named 'server.nvgpu'`

**解决**:
```bash
# 重新安装
cd /cu2tri
pip install -e .

# 或设置 PYTHONPATH
export PYTHONPATH=/cu2tri:$PYTHONPATH
```

## 性能优化

### 1. 调整并发任务数

在 GPU 配置文件中增加 `max_concurrent_tasks`：

```yaml
gpus:
  - name: "NVIDIA RTX 6000 Ada"
    max_concurrent_tasks: 20    # 增加并发数
```

### 2. 选择合适的负载均衡策略

```bash
# 轮询 (均匀分配)
nvgpu-server --gpu-config ... --lb-strategy round_robin

# 最少负载 (智能分配)
nvgpu-server --gpu-config ... --lb-strategy least_loaded
```

### 3. 调整内存阈值

```yaml
gpus:
  - name: "NVIDIA RTX 6000 Ada"
    memory_threshold: 0.85    # 允许更高内存使用
```

## 监控

### 实时监控 GPU 状态

```bash
# 使用 nvidia-smi 实时监控
watch -n 1 nvidia-smi

# 使用 NVGPU API
watch -n 1 'curl http://localhost:8080/api/v1/gpus'
```

### 查看任务队列

```bash
curl http://localhost:8080/api/v1/tasks
```

## 生产环境部署

### 使用 systemd

创建 `/etc/systemd/system/nvgpu-server.service`：

```ini
[Unit]
Description=NVGPU Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/cu2tri
Environment="PROJECT_ROOT=/cu2tri"
ExecStart=/usr/local/bin/nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl start nvgpu-server
sudo systemctl enable nvgpu-server
```

### 使用 Docker

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /cu2tri
COPY . .

RUN pip install -e ".[nvgpu]"

EXPOSE 8080

CMD ["nvgpu-server", "--gpu-config", "server/nvgpu/configs/gpu_resources/P250_A6000.yml"]
```
