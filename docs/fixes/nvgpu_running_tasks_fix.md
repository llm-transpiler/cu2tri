# NVGPU服务器 running_tasks 残留问题修复文档

## 问题描述

### 现象
- NVGPU服务器的GPU显示有10个`running_tasks`，但实际没有进程在运行
- 新任务一直处于`pending`状态，无法调度
- `nvidia-smi`显示GPU空闲，但服务器认为GPU已满

### 复现步骤
```bash
# 1. 启动NVGPU服务器并运行任务
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 2. 任务运行中，强制杀死服务器进程
kill -9 <PID>

# 3. 重新启动服务器
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 4. 查询GPU状态
curl http://localhost:8080/gpus | jq
# 结果: running_tasks 里仍有之前的任务ID
```

## 根本原因分析

### 代码逻辑
NVGPU使用daemon线程执行任务：

```python
# server/nvgpu/scheduler.py
def _schedule_round(self):
    """调度一轮任务"""
    for gpu_id, task in tasks_to_run:
        # 1. 标记任务为运行中
        self.gpu_manager.mark_task_running(gpu_id, task)

        # 2. 创建daemon线程执行任务
        thread = threading.Thread(
            target=self._execute_task,
            args=(task, gpu_id),
            daemon=True  # ← 关键: daemon线程
        )
        thread.start()

def _execute_task(self, task, gpu_id: int):
    """执行单个任务"""
    try:
        success = self.task_runner.run_task(task, gpu_id)
    finally:
        # 清理: 标记任务完成
        self.gpu_manager.mark_task_completed(gpu_id, task)
```

### 问题根源

**Daemon线程的特性**：当主线程退出时（服务器被kill、崩溃等），daemon线程会**立即终止**，不会等待`finally`块执行完成。

执行流程：
```
正常情况:
┌─────────────────────────────────────────────────────────┐
│ 1. mark_task_running() → running_tasks = [task_id]     │
│ 2. daemon线程执行任务                                     │
│ 3. 任务完成                                              │
│ 4. finally块执行 → mark_task_completed()                │
│    → running_tasks.remove(task_id) ✓                    │
└─────────────────────────────────────────────────────────┘

异常退出 (kill -9):
┌─────────────────────────────────────────────────────────┐
│ 1. mark_task_running() → running_tasks = [task_id]     │
│ 2. daemon线程执行任务                                     │
│ 3. 主进程被kill -9 立即终止!                             │
│ 4. daemon线程立即终止, finally块未执行 ✗                 │
│    → running_tasks 仍残留 task_id                        │
└─────────────────────────────────────────────────────────┘
```

### 历史代码对比

这个**不是重构引入的新bug**，而是原有设计问题：

```bash
# 重构前 (commit a2b91d7) 和重构后 (commit f93dbf5) 完全相同
git show a2b91d7:server/nvgpu/scheduler.py | grep -A 5 "daemon=True"
# 输出:
thread = threading.Thread(
    target=self._execute_task,
    args=(task, gpu_id),
    daemon=True  # ← 一直是这样
)
```

重构只改变了导入路径，核心逻辑完全一致：
```diff
- from gpu_manager import GPUManager
+ from server.nvgpu.gpu_manager import GPUManager
```

## 修复方案

### 方案选择

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| 改为非daemon线程 | finally保证执行 | 主线程退出时会阻塞 | ✗ 可能影响服务器关闭 |
| 持久化状态 | 完全恢复 | 复杂度高 | ✗ 过度设计 |
| **启动时清理** | **简单有效** | **治标不治本** | **✓ 采用** |

### 实现的修复

#### 1. 添加清理方法 (gpu_manager.py)

**新增方法**:
```python
# server/nvgpu/gpu_manager.py
def clear_stale_running_tasks(self):
    """Clear stale running tasks from all GPUs.

    This should be called on startup to clean up any tasks that were
    marked as running but didn't complete cleanly (e.g., server crash).
    """
    with self.lock:
        cleared_count = 0
        for gpu_id, gpu in self.gpus.items():
            if gpu.running_tasks:
                logger.warning(
                    f"Clearing {len(gpu.running_tasks)} stale running tasks from GPU {gpu_id}: {gpu.running_tasks}"
                )
                gpu.running_tasks.clear()
                cleared_count += len(gpu.running_tasks)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} stale running tasks from all GPUs")
```

#### 2. 启动时调用 (main.py)

**在服务器启动流程中添加清理调用**:
```python
# server/nvgpu/main.py - NVGPUServer.start() 方法
def start(self):
    """Start all server components."""
    logger.info("=" * 60)
    logger.info("Starting NVGPU Server")
    logger.info("=" * 60)

    # Auto-register GPUs from config if enabled
    if self.gpu_config_loader and self.gpu_config_loader.should_auto_register():
        logger.info("Auto-registering GPUs from configuration...")
        for gpu_config in self.gpu_config_loader.get_enabled_gpus():
            # ... 注册GPU

    # Start GPU monitoring
    self.gpu_manager.start_monitoring()

    # ← 新增: Clear any stale running tasks from previous server sessions
    self.gpu_manager.clear_stale_running_tasks()

    # Start scheduler
    self.scheduler.start()
    # ...
```

#### 3. 增强日志 (gpu_manager.py)

**提升mark_task_completed日志级别**，方便观察执行情况：
```python
# 修改前
def mark_task_completed(self, gpu_id: int, task: Task) -> bool:
    with self.lock:
        if gpu_id not in self.gpus:
            return False
        gpu = self.gpus[gpu_id]
        if task.task_id in gpu.running_tasks:
            gpu.running_tasks.remove(task.task_id)
            logger.debug(f"TASK {format_task_ref(task)} completed on GPU {gpu_id}")
        return True

# 修改后
def mark_task_completed(self, gpu_id: int, task: Task) -> bool:
    with self.lock:
        if gpu_id not in self.gpus:
            logger.warning(f"Cannot mark task completed: GPU {gpu_id} not found")
            return False

        gpu = self.gpus[gpu_id]
        if task.task_id in gpu.running_tasks:
            gpu.running_tasks.remove(task.task_id)
            logger.info(f"TASK {format_task_ref(task)} completed on GPU {gpu_id}, "
                       f"remaining running tasks: {len(gpu.running_tasks)}")
        else:
            logger.warning(
                f"TASK {format_task_ref(task)} not found in GPU {gpu_id} running_tasks. "
                f"Current running_tasks: {gpu.running_tasks}"
            )
        return True
```

## 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `server/nvgpu/gpu_manager.py` | 添加 `clear_stale_running_tasks()` 方法 |
| `server/nvgpu/gpu_manager.py` | 修改 `mark_task_completed()` 日志级别 DEBUG→INFO |
| `server/nvgpu/gpu_manager.py` | 添加任务ID不存在时的警告日志 |
| `server/nvgpu/main.py` | 在 `start()` 方法中调用清理方法 |

## 验证修复

### 预期日志输出

正常启动（无残留）:
```
2026-03-08 08:00:00 | gpu_manager | INFO | GPU monitoring started
2026-03-08 08:00:00 | scheduler   | INFO | Scheduler started
```

启动时清理残留:
```
2026-03-08 08:00:00 | gpu_manager | WARNING | Clearing 10 stale running tasks from GPU 0: [task_id1, task_id2, ...]
2026-03-08 08:00:00 | gpu_manager | WARNING | Clearing 10 stale running tasks from GPU 1: [task_id1, task_id2, ...]
2026-03-08 08:00:00 | gpu_manager | INFO    | Cleared 20 stale running tasks from all GPUs
2026-03-08 08:00:00 | scheduler   | INFO     | Scheduler started
```

任务完成时（新日志）:
```
2026-03-08 08:01:00 | gpu_manager | INFO | TASK [add_1_15_64|at@1|r@1]::[abc-123] completed on GPU 0, remaining running tasks: 9
```

### 验证步骤

```bash
# 1. 确保代码已更新
cd /cu2tri
git pull
git log --oneline -1
# 应该看到: 8e7bc6e fix: clear stale running_tasks on GPU manager startup...

# 2. 重启服务器
# (先停止旧进程)
nvgpu-server --gpu-config server/nvgpu/configs/gpu_resources/P250_A6000.yml

# 3. 检查日志，应该看到清理消息（如果有残留）
tail -50 /cu2tri/server/nvgpu/logs/nvgpu_server_*.log

# 4. 检查GPU状态
curl http://localhost:8080/gpus | jq '.gpus[].running_tasks'
# 应该看到: [] 或实际运行中的任务ID

# 5. 提交测试任务
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"script_path": "test.py"}'

# 6. 检查任务是否正常调度
curl http://localhost:8080/tasks | jq '.tasks[0].status'
# 应该看到: "running" 而不是 "pending"
```

## 相关提交

```
8e7bc6e fix: clear stale running_tasks on GPU manager startup and improve logging
```

## 未来改进建议

1. **改为非daemon线程** + 优雅关闭机制
2. **持久化状态**到文件，启动时恢复
3. **定期健康检查**，发现僵尸任务自动清理
