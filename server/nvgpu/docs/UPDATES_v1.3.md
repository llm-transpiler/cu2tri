# Version 1.3.0 - 日志改进 & 任务取消

## 🎯 核心改进

### 1. ✅ 带时间戳的日志文件

**之前**: 所有运行共用一个日志文件 `logs/nvgpu_server.log`，每次运行覆盖

**现在**: 每次运行创建独立的带时间戳日志文件

```
logs/
├── nvgpu_server_20251003_081441.log
├── nvgpu_server_20251003_082315.log
└── nvgpu_server_20251003_083527.log
```

**好处**:
- 保留历史日志
- 便于问题追踪
- 对比不同运行的行为

**实现**:
```python
# main.py
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"logs/nvgpu_server_{timestamp}.log"
logger = setup_logger("main", log_file=log_file)
```

### 2. ✅ 取消运行中的任务

**之前**: 只能取消 pending/queued 任务，运行中的任务无法停止

**现在**: 支持强制取消正在运行的任务

**两种取消模式**:

1. **普通取消** (`force=False`): 只取消待处理的任务
2. **强制取消** (`force=True`): 终止运行中的进程

**终止流程**:
1. 发送 SIGTERM (5秒超时)
2. 如果未响应，发送 SIGKILL

**使用示例**:

```python
# Python 客户端
client.cancel_task(task_id, force=True)

# REST API
curl -X POST "http://localhost:8080/tasks/{task_id}/cancel?force=true"

# 查看结果
result = client.get_task(task_id)
print(result.status)  # 'cancelled'
print(result.error_message)  # 'Cancelled by user (force)'
```

## 📦 修改的文件

### 核心实现

1. **main.py**
   - 添加时间戳到日志文件名
   - 传递 task_runner 给 API

2. **task_runner.py**
   - 使用 `subprocess.Popen` 而不是 `subprocess.run`
   - 保存运行中进程引用 `self.running_processes`
   - 新增 `kill_task()` 方法（graceful → forceful）

3. **task_queue.py**
   - 新增 `force_cancel_task()` 方法
   - 支持取消运行中任务

4. **api_server.py**
   - 修改 `POST /tasks/{task_id}/cancel` 支持 `force` 参数
   - 增加 `task_runner` 全局引用

5. **client.py**
   - `cancel_task()` 方法增加 `force` 参数

### 文档

6. **TASK_CANCELLATION.md** - 任务取消功能完整指南
7. **UPDATES_v1.3.md** - 本文档
8. **examples/example_cancel_running_task.py** - 使用示例

## 🔧 API 变更

### 修改的端点

```
POST /tasks/{task_id}/cancel?force=<bool>
```

**参数**:
- `force` (可选, 默认 false): 是否强制取消运行中任务

**响应**:
```json
{
  "success": true,
  "task_id": "abc-123",
  "forced": true
}
```

## 📊 使用场景

### 场景 1: 停止异常任务

```python
# 任务运行时间过长或行为异常
client.cancel_task(task_id, force=True)
```

### 场景 2: 批量取消

```python
# 清空所有运行中的任务
stats = client.get_statistics()
for task in stats['running_tasks']:
    client.cancel_task(task['id'], force=True)
```

### 场景 3: 超时处理

```python
try:
    result = client.wait_for_task(task_id, timeout=300)
except TimeoutError:
    client.cancel_task(task_id, force=True)
```

## ⚠️  注意事项

1. **数据安全**: 强制取消可能导致数据丢失
2. **资源清理**: SIGKILL 无法被捕获，进程无法清理
3. **GPU 资源**: GPU 显存会在进程终止后自动释放
4. **日志空间**: 每次运行创建新日志，注意磁盘空间

## 🔄 向后兼容

✅ **完全向后兼容**

- 默认行为不变（`force=False`）
- 旧代码继续工作
- 新功能可选使用

## 📖 文档

- 📘 [TASK_CANCELLATION.md](TASK_CANCELLATION.md) - 任务取消详细指南
- 📘 [LOG_HANDLING.md](LOG_HANDLING.md) - 日志处理指南
- 📘 [README.md](README.md) - 主文档

## 🎓 示例

```bash
# 运行取消任务示例
python examples/example_cancel_running_task.py

# 查看带时间戳的日志
ls -lh logs/nvgpu_server_*.log
```

## 🐛 Bug 修复

- 修复了 `scheduler.py` 中访问不存在的 `task.stderr` 属性的问题
- 现在从文件读取 stderr 内容进行 GPU 错误检测

## 版本信息

- 版本: 1.3.0
- 日期: 2025-10-03
- 基于: v1.2.0 (max_concurrent_tasks 功能)
