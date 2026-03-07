# nvgpu

NVGPU Server 完整使用指南。

## 快速开始

```bash
# 启动服务器
cd /cu2tri/server/nvgpu
python main.py --gpu-config configs/gpu_resources/P250_A6000.yml
```

## Python 客户端

```python
import sys
sys.path.insert(0, '/cu2tri/server/nvgpu')
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 提交任务
task_id = client.submit_task("test.py")

# 等待完成
result = client.wait_for_task(task_id)
```

## 三个参数

```python
client.submit_task(
    "test.py",
    task_mode="shared",      # GPU 行为: "exclusive" | "shared"
    task_type="functional",  # 业务分类: "functional" | "performance" | "both"
    task_label="xpiler/add_3_3_256"  # 自定义标签
)
```

### 智能默认值

| task_type | task_mode |
|-----------|-----------|
| `"functional"` | `"shared"` |
| `"performance"` | `"exclusive"` |
| `"both"` | `"exclusive"` |
| 未指定 | `"shared"` |

## GPU 配置

### 配置文件格式

```yaml
# configs/gpu_resources/your_config.yml
gpus:
  - logical_id: 0            # 系统逻辑 ID
    nvidia_smi_id: 0         # nvidia-smi 显示的 ID
    cuda_visible_id: 0       # CUDA_VISIBLE_DEVICES 值
    name: "GPU 名称"
    uuid: "GPU-xxx"           # nvidia-smi -L 获取
    enabled: true             # 是否启用
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 10

server:
  auto_register_gpus: true
```

### 获取 GPU UUID

```bash
nvidia-smi -L
```

## API 端点

### 任务管理
- `POST /tasks` - 提交任务
- `GET /tasks/{task_id}` - 获取任务状态
- `GET /tasks?status={status}` - 列出任务
- `POST /tasks/{task_id}/cancel?force={bool}` - 取消任务
- `GET /tasks/{task_id}/log?log_type={type}&offset={offset}&limit={limit}` - 获取日志

### GPU 管理
- `GET /gpus` - 列出所有 GPU
- `GET /gpus/{gpu_id}` - 获取 GPU 详情
- `PUT /gpus/{gpu_id}/status` - 设置状态
- `PUT /gpus/{gpu_id}/mode` - 设置模式
- `DELETE /gpus/{gpu_id}/mode` - 清除手动模式
- `POST /gpus/register` - 注册 GPU
- `POST /gpus/{gpu_id}/unregister` - 注销 GPU
- `POST /gpus/{gpu_id}/error` - 触发错误
- `POST /gpus/clear_error?gpu_id={id}` - 清除错误

### 其他
- `GET /health` - 健康检查
- `GET /stats` - 统计信息

## 日志位置

```
/cu2tri/server/nvgpu/logs/
├── nvgpu_server.log              # 历史日志
├── nvgpu_server_YYYYMMDD_HHMMSS.log  # 会话日志
└── tasks/
    ├── {task_id}.log       # 任务摘要
    ├── {task_id}.stdout    # 标准输出
    └── {task_id}.stderr    # 标准错误
```

## 常见问题

### 任务一直 pending

检查 GPU 是否注册：
```bash
# 方法 1: 使用配置文件
python main.py --gpu-config configs/gpu_resources/P250_A6000.yml

# 方法 2: 直接指定 GPU
python main.py --gpus 0 1
```

### 脚本找不到文件

设置 `work_dir`：
```python
client.submit_task(
    script_path="/workspace/test/script.py",
    work_dir="/workspace/test"
)
```

### GPU 处于 ERROR 状态

```bash
# 自动恢复（60秒后）
# 或手动清除
curl -X POST http://localhost:8080/gpus/clear_error
```
