# NVGPU Server API 参考手册

NVGPU Server 完整 API 参考，包含 Python 客户端示例和 curl 命令。

**版本:** v2.5 (最终版 - 概念分离设计)  
**基础 URL:** `http://localhost:8080` (默认)

---

## 目录

1. [概述](#概述)
2. [任务管理](#任务管理)
3. [GPU管理](#gpu管理)
4. [监控](#监控)
5. [错误处理](#错误处理)
6. [Python客户端](#python客户端)
7. [完整工作流](#完整工作流)

---

## 概述

### 核心概念

#### 三个参数，清晰分离

**1. task_mode - GPU 行为控制**
- **用途:** 控制 GPU 如何执行任务
- **取值:** `"exclusive"` 或 `"shared"`
- **默认:** 智能默认（基于 `task_type`）

**2. task_type - 业务分类**
- **用途:** 业务层面的分类，用于统计和筛选
- **取值:**
  - `"functional"` - 功能测试
  - `"performance"` - 性能测试
  - `"both"` - 两者兼有
- **默认:** `None`（可选）

**3. task_label - 具体标识**
- **用途:** 具体的测试标签，精确识别
- **取值:** 任意字符串（建议层次结构）
- **示例:** `"xpiler_cuda/add_3_3_256/cuda_vs_triton"`
- **默认:** `None`（可选）

#### 智能默认值

```
task_type="functional"   → task_mode="shared"
task_type="performance"  → task_mode="exclusive"
task_type="both"         → task_mode="exclusive" (包含性能测试)
task_type=None           → task_mode="shared" (安全默认)
```

#### GPU 模式

- `exclusive`: GPU 一次只接受一个任务
- `shared`: GPU 接受多个任务（受 `max_concurrent_tasks` 和 `memory_threshold` 控制）
- **手动模式:** 管理员可设置手动模式覆盖
- **任务驱动模式:** GPU 模式根据任务需求自动切换

### 基础 URL

```
http://localhost:8080
```

### 响应格式

所有响应都是 JSON 格式。

**成功响应:**
```json
{
  "success": true,
  ...
}
```

**错误响应:**
```json
{
  "detail": "错误信息"
}
```

---

## 任务管理

### 提交任务

提交新任务到服务器。

**端点:** `POST /tasks`

**请求体:**
```json
{
  "script_path": "/path/to/script.py",     // 必需
  "task_mode": "shared",                   // 可选 (智能默认)
  "task_type": "functional",               // 可选 (业务分类)
  "task_label": "xpiler_cuda/add_3_3_256", // 可选 (具体标识)
  "work_dir": ".",                         // 可选 (默认: ".")
  "args": ["--arg1", "value1"],            // 可选 (默认: [])
  "env": {"VAR": "value"},                 // 可选 (默认: null)
  "gpu_id": 0                              // 可选 (默认: null)
}
```

**参数说明:**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `script_path` | string | ✓ | - | Python 脚本路径 |
| `task_mode` | string | ✗ | 智能默认 | `"exclusive"` 或 `"shared"` |
| `task_type` | string | ✗ | `null` | 业务分类 (`"functional"`, `"performance"`, `"both"`) |
| `task_label` | string | ✗ | `null` | 具体标识（任意字符串） |
| `work_dir` | string | ✗ | `"."` | 工作目录 |
| `args` | array | ✗ | `[]` | 命令行参数 |
| `env` | object | ✗ | `null` | 环境变量 |
| `gpu_id` | integer | ✗ | `null` | 指定 GPU ID |

**智能默认值规则:**

如果 `task_mode` 未指定：
1. `task_type="functional"` → `task_mode="shared"`
2. `task_type="performance"` → `task_mode="exclusive"`
3. `task_type="both"` → `task_mode="exclusive"`
4. 其他情况 → `task_mode="shared"` (安全默认)

如果显式指定 `task_mode`，则使用指定值（覆盖智能默认）。

**响应:**
```json
{
  "success": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "task_mode": "shared",
    "task_type": "functional",
    "task_label": "xpiler_cuda/add_3_3_256",
    "script_path": "/path/to/script.py",
    "status": "pending",
    "submit_time": "2025-01-15T10:30:00.123456",
    ...
  }
}
```

**Python 客户端:**
```python
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 1. 最简单（90% 场景）
task_id = client.submit_task("test.py")

# 2. 功能测试（自动 shared）
task_id = client.submit_task(
    "test.py",
    task_type="functional"
)

# 3. 性能测试（自动 exclusive）
task_id = client.submit_task(
    "benchmark.py",
    task_type="performance"
)

# 4. 带具体标识
task_id = client.submit_task(
    "test_add.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)

# 5. 完全控制（覆盖默认）
task_id = client.submit_task(
    "memory_test.py",
    task_mode="exclusive",  # 显式指定
    task_type="functional",
    task_label="stress_test/memory_limit"
)
```

**curl 命令:**
```bash
# 最简单
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"script_path": "/workspace/test.py"}'

# 功能测试
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "script_path": "/workspace/test.py",
    "task_type": "functional"
  }'

# 完整示例
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "script_path": "/workspace/test.py",
    "task_type": "functional",
    "task_label": "xpiler_cuda/add_3_3_256/cuda_vs_triton",
    "args": ["--verbose"]
  }'
```

---

### 查询任务状态

获取任务的状态和结果。

**端点:** `GET /tasks/{task_id}`

**响应:**
```json
{
  "task_id": "550e8400-...",
  "task_mode": "shared",
  "task_type": "functional",
  "task_label": "xpiler_cuda/add_3_3_256",
  "script_path": "/path/to/script.py",
  "status": "completed",
  "assigned_gpu": 0,
  "submit_time": "2025-01-15T10:30:00.000",
  "queued_time": "2025-01-15T10:30:01.250",
  "start_time": "2025-01-15T10:30:04.650",
  "end_time": "2025-01-15T10:30:17.150",
  "exit_code": 0,
  "pending_time_ms": 1250.50,
  "queue_time_ms": 3400.25,
  "waiting_time_ms": 4650.75,
  "execution_time_ms": 12500.00,
  "total_time_ms": 17150.25
}
```

**时间字段说明（所有时间单位为毫秒，保留2位小数）:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `submit_time` | ISO时间 | 任务提交时间 |
| `queued_time` | ISO时间 | 分配到GPU队列的时间 |
| `start_time` | ISO时间 | 开始执行时间 |
| `end_time` | ISO时间 | 结束时间 |
| `pending_time_ms` | 浮点数 | PENDING 阶段时间（提交→分配GPU） |
| `queue_time_ms` | 浮点数 | QUEUED 阶段时间（分配GPU→开始执行） |
| `waiting_time_ms` | 浮点数 | 总等待时间（提交→开始执行） |
| `execution_time_ms` | 浮点数 | 执行时间（开始→结束） |
| `total_time_ms` | 浮点数 | 总时间（提交→结束） |

**时间线关系:**
```
submit ────→ queued ────→ start ────────→ end
    |          |            |               |
    |<-pending>|            |               |
    |          |<-queue---->|               |
    |<-----waiting--------->|               |
    |                       |<-execution--->|
    |<----------total-----------------------|
```

**任务状态:**
- `pending`: 等待中
- `running`: 运行中
- `completed`: 已完成
- `failed`: 失败
- `cancelled`: 已取消

**Python 客户端:**
```python
result = client.get_task(task_id)
print(f"Status: {result.status}")
print(f"Exit code: {result.exit_code}")

# 访问时间字段（毫秒，2位小数）
print(f"\n任务计时（毫秒）:")
print(f"  Pending时间:   {result.pending_time_ms:.2f} ms")
print(f"  Queue时间:     {result.queue_time_ms:.2f} ms")
print(f"  等待时间:      {result.waiting_time_ms:.2f} ms")
print(f "  执行时间:      {result.execution_time_ms:.2f} ms")
print(f"  总时间:        {result.total_time_ms:.2f} ms")
```

**curl 命令:**
```bash
curl http://localhost:8080/tasks/{task_id}
```

---

### 列出所有任务

获取任务列表（可选过滤）。

**端点:** `GET /tasks`

**查询参数:**
- `status`: 按状态过滤 (`pending`, `running`, `completed`, `failed`, `cancelled`)
- `task_type`: 按类型过滤 (`functional`, `performance`, `both`)
- `task_label`: 按标签过滤
- `gpu_id`: 按 GPU ID 过滤

**响应:**
```json
{
  "tasks": [
    {
      "task_id": "...",
      "task_mode": "shared",
      "task_type": "functional",
      "task_label": "xpiler_cuda/test1",
      "status": "completed",
      ...
    }
  ]
}
```

**Python 客户端:**
```python
# 所有任务
all_tasks = client.list_tasks()

# 按类型过滤
functional_tasks = client.list_tasks(task_type="functional")
perf_tasks = client.list_tasks(task_type="performance")

# 按标签过滤
cuda_tests = client.list_tasks(task_label="xpiler_cuda/add_3_3_256")

# 按状态过滤
running = client.list_tasks(status="running")

# 组合过滤
completed_functional = client.list_tasks(
    status="completed",
    task_type="functional"
)
```

**curl 命令:**
```bash
# 所有任务
curl http://localhost:8080/tasks

# 按类型过滤
curl "http://localhost:8080/tasks?task_type=functional"

# 组合过滤
curl "http://localhost:8080/tasks?status=completed&task_type=performance"
```

---

### 获取任务日志

获取任务的 summary/stdout/stderr 日志。

**端点:** `GET /tasks/{task_id}/log`

**查询参数:**
- `log_type`: 日志类型，`summary`（默认）、`stdout`、`stderr`
- `offset`: 起始字节偏移（默认: 0，必须 ≥ 0）
- `limit`: 返回最大字节数（默认: 102400）

**响应:**
```json
{
  "task_id": "550e8400-...",
  "log_type": "stdout",
  "log_path": "/workspace/server/nvgpu/logs/tasks/550e8400-....stdout",
  "content": "日志内容...",
  "total_size": 8192,
  "offset": 0,
  "size": 1024,
  "truncated": true,
  "has_more": true
}
```

**Python 客户端:**
```python
# 获取全部 stdout（自动分块）
full_log = client.get_full_task_log(task_id, "stdout")

# 按偏移/长度获取片段
chunk = client.get_task_log(task_id, "stderr", offset=0, limit=1024)
```

**curl 命令:**
```bash
# 获取 stdout 片段
curl "http://localhost:8080/tasks/{task_id}/log?log_type=stdout&offset=0&limit=1024"

# 获取 summary（默认）
curl "http://localhost:8080/tasks/{task_id}/log"
```

---

### 取消任务

取消正在运行或等待的任务。

**端点:** `POST /tasks/{task_id}/cancel`

**响应:**
```json
{
  "success": true,
  "message": "Task cancelled successfully"
}
```

**Python 客户端:**
```python
success = client.cancel_task(task_id)
```

**curl 命令:**
```bash
curl -X POST http://localhost:8080/tasks/{task_id}/cancel
```

---

## GPU管理

### 列出所有 GPU

获取所有 GPU 的状态信息。

**端点:** `GET /gpus`

**响应:**
```json
{
  "gpus": [
    {
      "gpu_id": 0,
      "mode": "shared",
      "manual_mode": null,
      "mode_locked_by": null,
      "status": "online",
      "memory_threshold": 0.75,
      "max_concurrent_tasks": 4,
      "current_memory_usage": 0.35,
      "running_tasks": ["task-id-1"],
      "running_task_count": 1,
      "error_message": null
    }
  ]
}
```

**GPU 状态:**
- `online`: 在线可用
- `offline`: 离线不可用
- `maintenance`: 维护中
- `error`: 错误状态

**Python 客户端:**
```python
gpus = client.list_gpus()
for gpu in gpus:
    print(f"GPU {gpu['gpu_id']}: {gpu['status']}, "
          f"mode={gpu['mode']}, tasks={gpu['running_task_count']}")
```

**curl 命令:**
```bash
curl http://localhost:8080/gpus
```

---

### 获取单个 GPU 信息

获取特定 GPU 的详细信息。

**端点:** `GET /gpus/{gpu_id}`

**响应:** 同上述 GPU 对象格式

**Python 客户端:**
```python
gpu = client.get_gpu(0)
print(f"GPU 0 memory: {gpu['current_memory_usage']*100:.1f}%")
```

**curl 命令:**
```bash
curl http://localhost:8080/gpus/0
```

---

### 设置 GPU 状态

设置 GPU 的状态（在线/离线/维护）。

**端点:** `PUT /gpus/{gpu_id}/status`

**请求体:**
```json
{
  "status": "online"  // "online", "offline", "maintenance"
}
```

**Python 客户端:**
```python
client.set_gpu_status(0, "maintenance")
```

**curl 命令:**
```bash
curl -X PUT http://localhost:8080/gpus/0/status \
  -H "Content-Type: application/json" \
  -d '{"status": "maintenance"}'
```

---

### 设置 GPU 模式（手动覆盖）

设置 GPU 的执行模式（手动覆盖）。

**端点:** `PUT /gpus/{gpu_id}/mode`

**请求体:**
```json
{
  "mode": "exclusive",  // "exclusive" 或 "shared"
  "manual": true        // true=手动模式覆盖，false=仅改变当前模式
}
```

**Python 客户端:**
```python
# 设置手动模式覆盖（持续到清除）
client.set_gpu_mode(0, "exclusive", manual=True)

# 仅改变当前模式（任务驱动）
client.set_gpu_mode(0, "exclusive", manual=False)
```

**curl 命令:**
```bash
curl -X PUT http://localhost:8080/gpus/0/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "exclusive", "manual": true}'
```

---

### 清除 GPU 手动模式

清除 GPU 的手动模式覆盖，恢复任务驱动模式切换。

**端点:** `DELETE /gpus/{gpu_id}/mode`

**响应:**
```json
{
  "success": true,
  "gpu_id": 0,
  "message": "Manual mode cleared"
}
```

**Python 客户端:**
```python
client.clear_gpu_manual_mode(0)
```

**curl 命令:**
```bash
curl -X DELETE http://localhost:8080/gpus/0/mode
```

---

### 配置 GPU 内存阈值

设置 GPU 的内存使用阈值，超过阈值将拒绝新任务。

**端点:** `PUT /gpus/{gpu_id}/memory_threshold`

**请求体:**
```json
{
  "threshold": 0.8
}
```

**Python 客户端:**
```python
# 通过 API 请求设置
import requests
requests.put(
    "http://localhost:8080/gpus/0/memory_threshold",
    json={"threshold": 0.8}
)
```

**curl 命令:**
```bash
curl -X PUT http://localhost:8080/gpus/0/memory_threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.8}'
```

---

### 配置 GPU 最大并发任务数

设置 shared 模式下 GPU 可同时运行的最大任务数。

**端点:** `PUT /gpus/{gpu_id}/max_concurrent_tasks`

**请求体:**
```json
{
  "max_tasks": 6
}
```

**Python 客户端:**
```python
# 通过 API 请求设置
import requests
requests.put(
    "http://localhost:8080/gpus/0/max_concurrent_tasks",
    json={"max_tasks": 6}
)
```

**curl 命令:**
```bash
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 6}'
```

---

## 监控

### 健康检查

检查服务器是否运行正常。

**端点:** `GET /health`

**响应:**
```json
{
  "status": "ok"
}
```

**Python 客户端:**
```python
if client.health_check():
    print("Server is healthy")
```

**curl 命令:**
```bash
curl http://localhost:8080/health
```

---

### 获取服务器统计

获取服务器队列与 GPU 的统计信息。

**端点:** `GET /stats`

**响应:**
```json
{
  "queue": {
    "global_queue_size": 5,
    "total_tasks": 128,
    "pending": 5,
    "queued": 2,
    "running": 3,
    "completed": 116,
    "failed": 2,
    "gpu_queues": {"0": 1, "1": 1}
  },
  "gpus": {
    "total_gpus": 2,
    "online_gpus": 2,
    "severe_error_active": false
  }
}
```

**Python 客户端:**
```python
stats = client.get_stats()
print(f"Pending: {stats['queue']['pending']}, Running: {stats['queue']['running']}")
```

**curl 命令:**
```bash
curl http://localhost:8080/stats
```

---

## 错误处理

### 触发严重错误（测试用）

手动触发 GPU 严重错误状态（仅用于测试）。

**端点:** `POST /gpus/{gpu_id}/error`

**请求体:**
```json
{
  "error_message": "Test error"
}
```

**效果:**
- GPU 状态变为 ERROR
- 所有运行中的任务被强制终止
- 被终止的任务重新排队（插入队列最前面）
- 系统暂停调度 60 秒后自动恢复

---

### 清除严重错误状态

手动清除严重错误状态。

**端点:** `POST /clear_severe_error`

**响应:**
```json
{
  "success": true
}
```

**Python 客户端:**
```python
client.clear_severe_error()
```

**curl 命令:**
```bash
curl -X POST http://localhost:8080/gpus/clear_error
```

---

## Python客户端

### 安装与初始化

```python
import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from server.nvgpu.client import NVGPUClient

# 初始化客户端
client = NVGPUClient("http://localhost:8080")

# 健康检查
if not client.health_check():
    raise RuntimeError("Server not available")
```

### 基本任务提交

```python
# 方式 1: 最简单（使用所有默认值）
task_id = client.submit_task("test.py")

# 方式 2: 功能测试（自动 shared）
task_id = client.submit_task("test.py", task_type="functional")

# 方式 3: 性能测试（自动 exclusive）
task_id = client.submit_task("benchmark.py", task_type="performance")

# 方式 4: 带具体标识
task_id = client.submit_task(
    "test_add.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)

# 方式 5: 完全控制（覆盖默认）
task_id = client.submit_task(
    "memory_test.py",
    task_mode="exclusive",  # 显式指定
    task_type="functional",
    task_label="stress_test/memory_limit",
    args=["--stress"],
    gpu_id=0
)

# 方式 6: 在脚本目录中执行
task_id = client.submit_task_in_script_dir(
    "/workspace/tests/my_test.py",
    task_type="functional"
)
```

### 等待任务完成

```python
import time

def wait_for_task(client, task_id, timeout=300):
    """等待任务完成"""
    start = time.time()
    while time.time() - start < timeout:
        result = client.get_task(task_id)
        
        if result.status in ['completed', 'failed', 'cancelled']:
            return result
        
        time.sleep(2)
    
    raise TimeoutError(f"Task {task_id} timeout")

# 使用
result = wait_for_task(client, task_id)
if result.status == 'completed' and result.exit_code == 0:
    print("✓ Task succeeded")
    stdout = client.get_full_task_log(task_id, "stdout")
else:
    print("✗ Task failed")
    stderr = client.get_full_task_log(task_id, "stderr")
```

### 批量任务提交

```python
# 提交多个任务
task_ids = []
for i in range(10):
    task_id = client.submit_task(
        f"test_{i}.py",
        task_type="functional",
        task_label=f"batch_test/test_{i}",
        args=["--test-id", str(i)]
    )
    task_ids.append(task_id)

# 等待所有完成
results = []
for task_id in task_ids:
    result = wait_for_task(client, task_id)
    results.append(result)

# 统计结果
success = sum(1 for r in results if r.status == 'completed' and r.exit_code == 0)
print(f"✓ {success}/{len(results)} tasks succeeded")
```

### GPU 管理示例

```python
# 查看所有 GPU
gpus = client.list_gpus()
for gpu in gpus:
    print(f"GPU {gpu['gpu_id']}: {gpu['status']}, "
          f"mode={gpu['mode']}, "
          f"running={gpu['running_task_count']}/{gpu['max_concurrent_tasks']}")

# 设置 GPU 为独占模式
client.set_gpu_mode(0, "exclusive", manual=True)

# 运行独占任务
task_id = client.submit_task(
    "exclusive_test.py",
    task_mode="exclusive",
    gpu_id=0
)
wait_for_task(client, task_id)

# 恢复共享模式
client.clear_gpu_manual_mode(0)
```

---

## 完整工作流

### 工作流 1: 功能测试

```python
#!/usr/bin/env python3
"""运行功能测试"""

from server.nvgpu.client import NVGPUClient
import time

client = NVGPUClient("http://localhost:8080")

# 提交功能测试（自动使用 shared 模式）
task_id = client.submit_task(
    "/workspace/tests/functional_test.py",
    task_type="functional",
    task_label="xpiler_cuda/test_add",
    args=["--verbose"]
)

print(f"Task submitted: {task_id[:8]}")

# 轮询状态
while True:
    result = client.get_task(task_id)
    print(f"Status: {result.status}...", end='\r')
    
    if result.status in ['completed', 'failed', 'cancelled']:
        break
    
    time.sleep(1)

# 检查结果
print(f"\nTask {result.status}")
print(f"Exit code: {result.exit_code}")
print(f"Execution time: {result.execution_time_ms}ms")

if result.exit_code == 0:
    print("✓ Test PASSED")
else:
    print("✗ Test FAILED")
    stderr = client.get_full_task_log(task_id, "stderr")
    print(stderr)
```

### 工作流 2: 性能基准测试

```python
#!/usr/bin/env python3
"""运行性能基准测试（独占 GPU）"""

from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 提交性能测试（自动使用 exclusive 模式）
task_id = client.submit_task(
    "/workspace/benchmarks/matmul_bench.py",
    task_type="performance",
    task_label="matmul/4096x4096/baseline",
    args=["--size", "4096", "--iterations", "100"],
    gpu_id=0
)

print(f"Benchmark task: {task_id[:8]}")
print("Waiting for completion...")

# 等待完成
result = wait_for_task(client, task_id, timeout=600)

# 解析结果
if result.exit_code == 0:
    stdout = client.get_full_task_log(task_id, "stdout")
    # 解析性能数据
    for line in stdout.split('\n'):
        if 'TFLOPS' in line:
            print(line)
```

### 工作流 3: 批量测试

```python
#!/usr/bin/env python3
"""批量运行测试套件"""

from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 定义测试列表
tests = [
    ("test1.py", "xpiler_cuda/test1"),
    ("test2.py", "xpiler_cuda/test2"),
    ("test3.py", "xpiler_cuda/test3"),
]

# 提交所有测试
task_ids = []
for script, label in tests:
    task_id = client.submit_task(
        script,
        task_type="functional",
        task_label=label
    )
    task_ids.append((label, task_id))
    print(f"Submitted: {label} - {task_id[:8]}")

# 等待所有完成
results = []
for label, task_id in task_ids:
    result = wait_for_task(client, task_id)
    results.append((label, result))
    print(f"{label}: {result.status}")

# 统计结果
success = sum(1 for _, r in results if r.exit_code == 0)
print(f"\n✓ {success}/{len(results)} tests passed")
```

---

## 最佳实践

### 1. 任务分类

使用 `task_type` 进行业务分类：
```python
task_type="functional"   # 功能测试
task_type="performance"  # 性能测试
task_type="both"         # 综合测试
```

### 2. 标签命名

使用层次结构的标签：
```python
# 推荐
task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
task_label="benchmark/matmul/4096x4096/fp32"
task_label="nightly_regression/test_suite_1/case_001"

# 避免
task_label="test"  # 太模糊
```

### 3. 选择正确的 task_mode

**使用 shared 当:**
- 任务 GPU 使用率低
- 任务可以并发运行
- 希望提高 GPU 利用率

**使用 exclusive 当:**
- 性能基准测试
- 任务需要全部 GPU 资源
- 需要稳定的性能结果

### 4. 错误处理

```python
try:
    task_id = client.submit_task("test.py")
    result = wait_for_task(client, task_id)
    
    if result.exit_code != 0:
        stderr = client.get_full_task_log(task_id, "stderr")
        raise RuntimeError(f"Task failed: {stderr}")
        
except TimeoutError:
    client.cancel_task(task_id)
    raise
```

### 5. GPU 资源管理

```python
# 检查 GPU 可用性
gpus = client.list_gpus()
available = [g for g in gpus if g['status'] == 'online']

if not available:
    raise RuntimeError("No GPUs available")

# 选择负载最低的 GPU
best_gpu = min(available, key=lambda g: g['running_task_count'])
task_id = client.submit_task("test.py", gpu_id=best_gpu['gpu_id'])
```

---

## 附录

### HTTP 状态码

- `200`: 成功
- `400`: 请求错误（参数无效）
- `404`: 资源未找到（任务 ID 或 GPU ID 不存在）
- `500`: 服务器内部错误

### 任务状态转换

```
pending → running → completed
                 → failed
                 → cancelled
```

### GPU 模式优先级

1. **手动模式** (manual_mode): 最高优先级，直到清除
2. **任务锁定** (mode_locked_by): exclusive 任务运行时
3. **默认模式**: shared

### 智能默认值决策树

```
是否指定 task_mode?
├─ 是 → 使用指定的 task_mode
└─ 否 → 检查 task_type
    ├─ task_type="functional" → task_mode="shared"
    ├─ task_type="performance" → task_mode="exclusive"
    ├─ task_type="both" → task_mode="exclusive"
    └─ 其他 → task_mode="shared" (安全默认)
```

---

## 相关文档

- [快速入门](QUICKSTART.md)
- [v2.5 最终设计](V2.5_FINAL_DESIGN.md)
- [v2.5 变更说明](V2.5_CHANGES.md)
- [文档中心](README.md)

---

**最后更新:** 2025-01-15  
**版本:** v2.5 (最终版)

