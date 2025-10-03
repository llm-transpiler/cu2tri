# 任务取消功能

## 概述

支持取消正在运行的任务，特别是异常运行或需要紧急停止的任务。

## 功能特性

### 1. 普通取消 (Normal Cancel)
- 只能取消 **pending** 或 **queued** 状态的任务
- 不会杀死正在运行的进程
- 安全且快速

### 2. 强制取消 (Force Cancel)
- 可以取消**任何状态**的任务，包括正在运行的
- 会终止正在运行的进程（graceful termination → SIGKILL）
- 适用于异常任务或紧急停止

## API 使用

### REST API

```bash
# 普通取消（只取消待处理的任务）
curl -X POST http://localhost:8080/tasks/{task_id}/cancel

# 强制取消（包括正在运行的任务）
curl -X POST "http://localhost:8080/tasks/{task_id}/cancel?force=true"
```

响应：
```json
{
  "success": true,
  "task_id": "abc-123",
  "forced": true
}
```

### Python 客户端

```python
from client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 普通取消
client.cancel_task(task_id, force=False)

# 强制取消
client.cancel_task(task_id, force=True)
```

## 终止流程

强制取消使用两阶段终止：

1. **Graceful Termination** (5秒超时)
   - 发送 SIGTERM 信号
   - 允许进程清理资源
   - 等待最多 5 秒

2. **Force Kill** (如果 graceful 失败)
   - 发送 SIGKILL 信号
   - 立即终止进程
   - 无法被捕获或忽略

## 使用场景

### 场景 1: 取消排队任务

```python
# 任务还在队列中，使用普通取消
task_id = client.submit_task(...)
client.cancel_task(task_id, force=False)
```

### 场景 2: 停止失控任务

```python
# 任务运行时间过长或行为异常
client.cancel_task(task_id, force=True)
```

### 场景 3: 批量取消

```python
# 取消多个运行中的任务
task_ids = [...]
for task_id in task_ids:
    client.cancel_task(task_id, force=True)
```

### 场景 4: 异常处理

```python
try:
    result = client.wait_for_task(task_id, timeout=300)
except TimeoutError:
    # 超时了，强制取消
    print("Task timed out, force cancelling...")
    client.cancel_task(task_id, force=True)
```

## 任务状态变化

### 普通取消
```
pending → cancelled
queued → cancelled
running → (失败，无法取消)
```

### 强制取消
```
pending → cancelled
queued → cancelled
running → cancelled (进程被终止)
completed → (失败，已完成的任务无法取消)
```

## 日志记录

任务被强制取消时的日志：

```
2025-10-03 08:30:15 | task_runner     | INFO     | Terminating task abc-123...
2025-10-03 08:30:15 | task_runner     | INFO     | Task abc-123 terminated gracefully
2025-10-03 08:30:15 | task_queue      | INFO     | Task abc-123 force cancelled
```

如果需要强制杀死：

```
2025-10-03 08:30:15 | task_runner     | INFO     | Terminating task abc-123...
2025-10-03 08:30:20 | task_runner     | WARNING  | Task abc-123 did not terminate, sending SIGKILL...
2025-10-03 08:30:20 | task_runner     | INFO     | Task abc-123 killed forcefully
```

## 任务日志

被取消的任务会在日志中标记：

```
=== Result ===
Status: cancelled
Exit Code: -15 (or -9 for SIGKILL)
Signal: SIGTERM (or SIGKILL)
Error Message: Cancelled by user (force)
```

## 注意事项

1. **数据丢失风险**: 强制取消会立即终止进程，可能导致未保存的数据丢失
2. **资源清理**: 被 SIGKILL 的进程无法执行清理代码
3. **GPU 内存**: 进程终止后 GPU 显存会自动释放
4. **并发任务**: 在 shared 模式下，其他任务可以继续运行

## 最佳实践

1. **优先使用普通取消**: 对于排队任务，使用 `force=False`
2. **确认再强制**: 强制取消前确认任务确实需要停止
3. **日志检查**: 取消后检查日志了解任务终止原因
4. **错误处理**: 在客户端代码中妥善处理取消失败的情况

## 示例

完整示例请参见：`examples/example_cancel_running_task.py`

```bash
python examples/example_cancel_running_task.py
```
