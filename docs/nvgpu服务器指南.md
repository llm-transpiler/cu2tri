# NVGPU 服务器指南

## 概述

NVGPU 服务器是一个基于 FastAPI 的 GPU 任务调度系统，支持多 GPU 资源管理和任务并发执行。

## 功能特性

- **GPU 资源管理**: 支持多 GPU 卡的资源分配和监控
- **任务调度**: 基于优先级的任务队列调度，支持 two-phase 调度
- **REST API**: 提供完整的 HTTP API 接口
- **任务执行**: 异步任务执行，支持超时和取消
- **状态监控**: 实时任务状态和 GPU 使用情况查询
- **负载均衡**: 支持多种负载均衡策略 (round_robin, least_loaded, fill)
- **GPU 模式**: 支持 shared (共享) 和 exclusive (独占) 两种模式
- **错误恢复**: 严重错误自动恢复机制 (60秒)

## 安装

```bash
cd /cu2tri
pip install -e ".[nvgpu]"
```

## GPU 配置

### 自动检测并生成配置 (推荐)

使用 `gpu-detect` 工具自动检测系统 GPU 并生成配置文件：

```bash
# 检测 GPU 并生成配置
cd /cu2tri
python -m server.nvgpu.tools.gpu_detect --print-command

# 自定义配置参数
python -m server.nvgpu.tools.gpu_detect \
    --mode exclusive \
    --max-tasks 5 \
    --memory-threshold 0.85 \
    --print-command

# 预览配置（不写入文件）
python -m server.nvgpu.tools.gpu_detect --dry-run
```

### 手动创建 GPU 配置文件

GPU 配置文件位于 `server/nvgpu/configs/gpu_resources/`，示例：

```yaml
# my_gpu.yml
gpus:
  - logical_id: 0                  # 系统逻辑 ID
    nvidia_smi_id: 0               # nvidia-smi 显示的 ID
    cuda_visible_id: 0             # CUDA_VISIBLE_DEVICES 映射
    name: "NVIDIA RTX 6000 Ada Generation"
    uuid: "GPU-1b8c43a8-..."       # nvidia-smi -L 获取
    memory_gb: 49.14               # GPU 内存大小
    enabled: true                  # 是否启用
    default_mode: "shared"         # exclusive 或 shared
    memory_threshold: 0.75         # 内存使用阈值 (0-1)
    max_concurrent_tasks: 10       # 最大并发任务数

server:
  auto_register_gpus: true         # 启动时自动注册 GPU
```

### 获取 GPU UUID

```bash
nvidia-smi -L
# 输出示例:
# GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-1b8c43a8-35ba-a01b-a2ee-0d6fe8f1f60f)
# GPU 1: NVIDIA RTX 6000 Ada Generation (UUID: GPU-78b50b8e-ef33-b3fd-2c52-e3f135990f31)
```

### 负载均衡策略

| 策略 | 说明 |
|------|------|
| `round_robin` | 轮询分配任务 |
| `least_loaded` | 分配给负载最轻的 GPU |
| `fill` | 优先填满第一个 GPU (默认) |

## 启动服务器

**重要**: 始终从项目根目录 `/cu2tri` 启动服务器

```bash
# 从项目根目录
cd /cu2tri
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml

# 自定义端口
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml --port 8848

# 指定日志级别
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml --log-level DEBUG

# 直接指定 GPU ID (简单场景)
nvgpu-server --gpus 0 1
```

### Python 模块方式

```bash
cd /cu2tri
python -m server.nvgpu.main --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml
```

## API 接口

服务器启动后访问 `http://localhost:8080` 或 `http://localhost:8080/docs` 查看自动生成的 API 文档。

**注意**: API 端点没有 `/api/v1/` 前缀

### 1. 健康检查

```bash
curl http://localhost:8080/health
```

### 2. 提交任务

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "script_path": "test.py",
    "task_mode": "shared",
    "task_type": "functional",
    "task_label": "my_test",
    "work_dir": "/path/to/project"
  }'
```

**任务参数说明**:

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `script_path` | 脚本路径 | - | 必填 |
| `task_mode` | GPU行为模式 | `exclusive`, `shared` | 根据task_type智能默认 |
| `task_type` | 业务分类 | `functional`, `performance`, `both` | - |
| `task_label` | 自定义标签 | 任意字符串 | - |
| `work_dir` | 工作目录 | - | `.` |
| `args` | 命令行参数 | `["--arg1", "value1"]` | `[]` |
| `env` | 环境变量 | `{"KEY": "value"}` | - |
| `gpu_id` | 指定 GPU ID | 整数 | - |

**智能默认值**:

| task_type | task_mode |
|-----------|-----------|
| `functional` | `shared` |
| `performance` | `exclusive` |
| `both` | `exclusive` |
| 未指定 | `shared` |

### 3. 查询任务状态

```bash
# 查询特定任务
curl http://localhost:8080/tasks/{task_id}

# 列出所有任务（可按状态筛选）
curl http://localhost:8080/tasks
curl http://localhost:8080/tasks?status=running
curl http://localhost:8080/tasks?status=pending
```

### 4. 取消任务

```bash
# 优雅取消（发送 SIGTERM）
curl -X POST http://localhost:8080/tasks/{task_id}/cancel

# 强制取消（发送 SIGKILL）
curl -X POST "http://localhost:8080/tasks/{task_id}/cancel?force=true"
```

### 5. 获取任务日志

```bash
# 获取日志摘要
curl http://localhost:8080/tasks/{task_id}/log

# 指定日志类型
curl "http://localhost:8080/tasks/{task_id}/log?log_type=summary"
curl "http://localhost:8080/tasks/{task_id}/log?log_type=stdout"
curl "http://localhost:8080/tasks/{task_id}/log?log_type=stderr"

# 分页获取
curl "http://localhost:8080/tasks/{task_id}/log?offset=0&limit=10240"
```

### 6. 获取 GPU 状态

```bash
# 列出所有 GPU
curl http://localhost:8080/gpus

# 获取特定 GPU 详情
curl http://localhost:8080/gpus/0
```

### 7. 管理 GPU

```bash
# 注册 GPU
curl -X POST http://localhost:8080/gpus/register \
  -H "Content-Type: application/json" \
  -d '{"gpu_id": 0, "mode": "shared"}'

# 注销 GPU
curl -X POST http://localhost:8080/gpus/0/unregister

# 设置 GPU 模式
curl -X PUT http://localhost:8080/gpus/0/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "exclusive", "manual": true}'

# 清除手动模式
curl -X DELETE http://localhost:8080/gpus/0/mode

# 设置内存阈值
curl -X PUT http://localhost:8080/gpus/0/memory_threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.85}'

# 设置最大并发任务数
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 10}'

# 触发错误状态
curl -X POST http://localhost:8080/gpus/0/error \
  -H "Content-Type: application/json" \
  -d '{"error_message": "Test error"}'

# 清除错误状态
curl -X POST http://localhost:8080/gpus/clear_error
```

### 8. 统计信息

```bash
curl http://localhost:8080/stats
```

## Python 客户端

```python
import sys
sys.path.insert(0, '/cu2tri')
from server.nvgpu.client import NVGPUClient

# 创建客户端
client = NVGPUClient("http://localhost:8080")

# 健康检查
is_healthy = client.health_check()
print(f"Server healthy: {is_healthy}")

# 提交任务
task_id = client.submit_task(
    script_path="train.py",
    task_mode="shared",       # GPU 行为: "exclusive" | "shared"
    task_type="performance",  # 业务分类: "functional" | "performance" | "both"
    task_label="xpiler/add_3_3_256",  # 自定义标签
    work_dir="/path/to/project",
    args=["--epochs", "10"],
    env={"CUDA_LAUNCH_BLOCKING": "1"}
)
print(f"Task submitted: {task_id}")

# 查询状态
status = client.get_task_status(task_id)
print(status)

# 等待完成
result = client.wait_for_task(task_id, timeout=3600, poll_interval=1)
print(f"Result: {result}")

# 获取日志
log_data = client.get_task_log(task_id, log_type="summary")
print(log_data["content"])

# 获取完整日志（自动分页）
full_log = client.get_full_task_log(task_id, log_type="stdout")

# 取消任务
client.cancel_task(task_id)
client.cancel_task(task_id, force=True)

# 设置 GPU 参数
client.set_gpu_memory_threshold(0, 0.85)
client.set_gpu_max_concurrent_tasks(0, 10)
```

## 任务时间指标

任务对象包含以下时间属性（单位：毫秒）：

| 属性 | 说明 |
|------|------|
| `pending_duration_ms` | 任务在全局队列等待时间 |
| `queue_duration_ms` | 任务在 GPU 本地队列等待时间 |
| `waiting_duration_ms` | pending + queue 总时间 |
| `running_duration_ms` | 任务实际执行时间 |
| `total_duration_ms` | 从提交到完成的总时间 |

## 日志

### 日志位置

```
/cu2tri/server/nvgpu/logs/
├── nvgpu_server.log                          # 历史日志 (追加)
├── nvgpu_server_YYYYMMDD_HHMMSS.log          # 当前会话日志
└── tasks/
    ├── {task_id}.log       # 任务摘要日志
    ├── {task_id}.stdout    # 任务标准输出
    └── {task_id}.stderr    # 任务标准错误
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
# 实时查看历史日志
tail -f /cu2tri/server/nvgpu/logs/nvgpu_server.log

# 查看特定任务日志
cat /cu2tri/server/nvgpu/logs/tasks/{task_id}.log
```

## 故障排除

### 1. 任务一直 pending

**原因**: 没有注册 GPU

**解决**:
```bash
# 检查可用 GPU
nvidia-smi --list-gpus

# 使用配置文件
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml

# 或直接指定 GPU
nvgpu-server --gpus 0 1
```

### 2. GPU 不可用

**问题**: `pynvml not available, GPU monitoring disabled`

**解决**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 安装 nvidia-ml-py3
pip install nvidia-ml-py3

# 检查 GPU 设备
ls /dev/nvidia*
```

### 3. 端口被占用

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

### 4. 权限问题

**问题**: `Permission denied` 访问 GPU

**解决**:
```bash
# 添加用户到 video 组
sudo usermod -a -G video $USER

# 重新登录生效
```

### 5. GPU 处于 ERROR 状态

**问题**: GPU 标记为 ERROR，不接受新任务

**解决**:
```bash
# 自动恢复 (60秒后)

# 或手动清除
curl -X POST http://localhost:8080/gpus/clear_error
```

## 性能优化

### 1. 调整并发任务数

在 GPU 配置文件中增加 `max_concurrent_tasks`：

```yaml
gpus:
  - name: "NVIDIA RTX 6000 Ada"
    max_concurrent_tasks: 20    # 增加并发数
```

或通过 API 动态调整：
```bash
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 20}'
```

### 2. 选择合适的负载均衡策略

```bash
# 轮询 (均匀分配)
nvgpu-server --gpu-config ... --lb-strategy round_robin

# 最少负载 (智能分配)
nvgpu-server --gpu-config ... --lb-strategy least_loaded

# 填充优先 (默认)
nvgpu-server --gpu-config ... --lb-strategy fill
```

### 3. 调整内存阈值

```yaml
gpus:
  - name: "NVIDIA RTX 6000 Ada"
    memory_threshold: 0.85    # 允许更高内存使用
```

或通过 API：
```bash
curl -X PUT http://localhost:8080/gpus/0/memory_threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.85}'
```

### 4. 选择合适的 GPU 模式

- **shared (共享)**: 多个任务可同时运行在同一 GPU 上，适合功能测试
- **exclusive (独占)**: GPU 只能运行一个任务，适合性能测试

## 监控

### 实时监控 GPU 状态

```bash
# 使用 nvidia-smi 实时监控
watch -n 1 nvidia-smi

# 使用 NVGPU API (注意: 没有 /api/v1/ 前缀)
watch -n 1 'curl http://localhost:8080/gpus'

# 查看服务器统计
watch -n 1 'curl http://localhost:8080/stats'
```

### 查看任务队列

```bash
# 查看所有任务
curl http://localhost:8080/tasks

# 按状态筛选
curl http://localhost:8080/tasks?status=pending
curl http://localhost:8080/tasks?status=running
curl http://localhost:8080/tasks?status=completed
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
ExecStart=/usr/local/bin/nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/my_gpu.yml
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

CMD ["nvgpu-server", "--gpu-config", "server/nvgpu/configs/gpu_resources/my_gpu.yml"]
```

运行容器：

```bash
docker run --gpus all -p 8080:8080 nvgpu-server
```
