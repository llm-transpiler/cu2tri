# 任务日志处理指南

## 概述

NVGPU Server 提供了优雅的日志处理机制，可以处理大型日志文件（包括 stdout 和 stderr），并能捕获各种异常情况（如 segmentation fault、core dump 等）。

## 特性

### 1. 分离式日志存储

任务日志分为三个独立文件：

- **Summary Log** (`{task_id}.log`): 任务摘要信息
  - 任务元数据（脚本路径、GPU、命令等）
  - 时间信息（提交、开始、结束、持续时间）
  - 执行结果（退出码、信号、状态）
  - stdout/stderr 的预览（小文件全部包含，大文件包含前 5KB）

- **STDOUT** (`{task_id}.stdout`): 完整的标准输出
  - 实时写入，无大小限制
  - 直接捕获子进程的 stdout

- **STDERR** (`{task_id}.stderr`): 完整的标准错误输出
  - 实时写入，无大小限制
  - 直接捕获子进程的 stderr

### 2. 异常情况处理

系统能够识别和报告各种异常情况：

- **正常退出** (exit_code = 0)
- **错误退出** (exit_code > 0)
- **信号终止** (exit_code < 0)
  - SIGSEGV (-11): Segmentation fault
  - SIGABRT (-6): Abort signal
  - SIGKILL (-9): Kill signal
  - 其他信号
- **超时** (Timeout)
- **异常** (Exception)

### 3. 绝对路径输出

任务开始时，日志会输出脚本的绝对路径，方便调试和追踪。

## API 使用

### 获取任务信息

```python
from client import NVGPUClient

client = NVGPUClient("http://localhost:8080")
result = client.get_task(task_id)

print(f"STDOUT 大小: {result.stdout_size} bytes")
print(f"STDERR 大小: {result.stderr_size} bytes")
print(f"日志文件: {result.log_file}")
```

### 获取日志内容

#### 方法 1: 分段获取（推荐用于大文件）

```python
# 获取 summary log
summary = client.get_task_log(task_id, log_type="summary")
print(summary['content'])

# 获取 stdout（前 100KB）
stdout = client.get_task_log(task_id, log_type="stdout", limit=102400)
print(f"STDOUT 预览: {stdout['content']}")
print(f"是否有更多内容: {stdout['has_more']}")

# 获取 stderr（从 offset 开始）
stderr = client.get_task_log(task_id, log_type="stderr", offset=1000, limit=5000)
```

#### 方法 2: 获取完整日志

```python
# 自动分块下载完整日志
full_stdout = client.get_full_task_log(task_id, log_type="stdout")
full_stderr = client.get_full_task_log(task_id, log_type="stderr")
```

### REST API

#### 获取日志内容

```bash
# 获取 summary
curl "http://localhost:8080/tasks/{task_id}/log?log_type=summary"

# 获取 stdout（前 10KB）
curl "http://localhost:8080/tasks/{task_id}/log?log_type=stdout&limit=10240"

# 获取 stderr（从 offset 5000 开始）
curl "http://localhost:8080/tasks/{task_id}/log?log_type=stderr&offset=5000&limit=10240"
```

响应格式：

```json
{
  "task_id": "abc-123",
  "log_type": "stdout",
  "log_path": "/workspace/server/nvgpu/logs/tasks/abc-123.stdout",
  "content": "...",
  "total_size": 1048576,
  "offset": 0,
  "size": 10240,
  "truncated": true,
  "has_more": true
}
```

## 日志示例

### Summary Log 示例

```
=== Task abc-123-def-456 ===
Type: functional
Script: /workspace/server/nvgpu/test_scripts/simple_functional_test.py
Work Dir: /workspace/server/nvgpu
GPU: 0
Command: /usr/bin/python3 /workspace/server/nvgpu/test_scripts/simple_functional_test.py

=== Timing ===
Submit Time: 2025-10-03 07:36:44.123456
Start Time: 2025-10-03 07:36:45.234567
End Time: 2025-10-03 07:36:54.345678
Duration: 9.11s

=== Result ===
Status: completed
Exit Code: 0

=== Output Files ===
STDOUT: /workspace/server/nvgpu/logs/tasks/abc-123.stdout (1.2KB)
STDERR: /workspace/server/nvgpu/logs/tasks/abc-123.stderr (0B)

=== STDOUT Preview ===
Testing simple CUDA operations...
✓ Computation successful
Result: 42.0

=== STDERR ===
(empty)

=== END OF LOG ===
```

### Segmentation Fault 示例

```
=== Result ===
Status: failed
Exit Code: -11
Signal: SIGSEGV
Note: Process was terminated by signal (possible segfault or core dump)
```

## 最佳实践

### 1. 处理大文件

对于大型输出（例如 > 100MB），使用分页获取：

```python
def process_large_log(client, task_id, log_type):
    offset = 0
    chunk_size = 1024 * 1024  # 1MB
    
    while True:
        chunk = client.get_task_log(task_id, log_type, offset, chunk_size)
        
        # 处理当前块
        process_chunk(chunk['content'])
        
        if not chunk['has_more']:
            break
        
        offset += chunk['size']
```

### 2. 检查日志大小

在下载前检查大小：

```python
result = client.get_task(task_id)

if result.stdout_size > 100 * 1024 * 1024:  # > 100MB
    print("警告：STDOUT 非常大，建议分块处理")
    # 使用分页
else:
    # 可以直接获取全部
    stdout = client.get_full_task_log(task_id, "stdout")
```

### 3. 错误诊断

```python
result = client.get_task(task_id)

if result.exit_code:
    if result.exit_code < 0:
        print(f"进程被信号终止: {result.error_message}")
        # 可能是 segfault，检查 stderr
        stderr = client.get_task_log(task_id, "stderr", limit=10240)
        print(stderr['content'])
    else:
        print(f"进程退出错误码: {result.exit_code}")
```

## 文件位置

所有日志文件位于：

```
/workspace/server/nvgpu/logs/tasks/
├── {task_id}.log       # Summary
├── {task_id}.stdout    # 完整 stdout
└── {task_id}.stderr    # 完整 stderr
```

## 性能考虑

- **内存效率**: stdout/stderr 直接写入文件，不占用内存
- **实时性**: 使用行缓冲，日志实时写入
- **网络效率**: 支持分页，避免传输大量数据
- **存储**: 日志文件永久保留，需定期清理旧日志

## 清理日志

手动清理旧日志：

```bash
# 删除 7 天前的日志
find /workspace/server/nvgpu/logs/tasks/ -type f -mtime +7 -delete
```

## 参考

- 完整示例：`examples/example_log_handling.py`
- 客户端 API：`client.py`
- 服务器实现：`task_runner.py`, `api_server.py`

