## GPUManager 设计文档（锁与监控线程）

本设计文档描述 `server/nvgpu/gpu_manager.py` 中 `GPUManager` 的并发模型与监控线程设计，重点阐述 `self.lock` 的使用场景与 `monitor_thread` 的生命周期管理，以及与 NVML 采样/调度的交互。

### 背景
- `GPUManager` 负责：
  - 维护 `self.gpus: dict[int, GPU]` 的注册表与状态（ONLINE/ERROR 等）。
  - 跟踪每张 GPU 的运行任务列表、内存使用率、模式（SHARED/EXCLUSIVE/手动模式）。
  - 提供调度接口选择可用 GPU。
  - 运行后台监控线程定期采样 GPU 内存与处理严重错误的暂停/恢复。

### 并发与锁（self.lock = threading.RLock）
`self.lock` 是可重入锁，用来保护对共享状态（`self.gpus`、严重错误标志位等）的读写一致性。所有对注册表与关键状态的增删改查都在 `with self.lock` 中完成。

核心使用点（节选）：

1) 注册/反注册 GPU（修改 `self.gpus`）
```71:86:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id in self.gpus:
        logger.warning(f"GPU {gpu_id} already registered")
        return False
    # ... 创建 GPU 并写入 self.gpus ...
    self.gpus[gpu_id] = gpu
    return True
```
```90:102:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id not in self.gpus:
        return False
    # ... 校验无运行任务 ...
    del self.gpus[gpu_id]
    return True
```

2) 状态/模式修改（保持状态一致性）
```106:112:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id not in self.gpus:
        return False
    self.gpus[gpu_id].status = status
    return True
```
```125:136:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id not in self.gpus:
        return False
    gpu = self.gpus[gpu_id]
    gpu.mode = mode
    if manual:
        gpu.manual_mode = mode
    return True
```

3) 任务运行/完成标记（修改 `running_tasks`）
```316:336:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id not in self.gpus:
        return False
    self.gpus[gpu_id].running_tasks.append(task_id)
    # 完成时移除 task_id
```

4) 严重错误流转（原子性地切换标志、复制任务列表、清理）
```345:360:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    self.severe_error_active = True
    self.severe_error_time = datetime.now()
    # 标记 GPU ERROR、复制 running_tasks 用于锁外处理
```
```386:390:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id in self.gpus:
        self.gpus[gpu_id].running_tasks.clear()
```
```395:399:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    self.severe_error_active = False
    self.severe_error_time = None
```

5) 查询/枚举与调度（读也加锁，避免读到不一致数据）
```262:270:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    return self.gpus.get(gpu_id)

with self.lock:
    return list(self.gpus.values())
```
```302:311:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    gpu_ids = list(self.gpus.keys())
# 锁外逐张刷新内存后
with self.lock:
    for gpu_id, gpu in self.gpus.items():
        if gpu.can_accept_task(task_mode):
            return gpu_id
```

6) 写入采样的内存使用率（采样在锁外，回写在锁内）
```417:420:/workspace/server/nvgpu/gpu_manager.py
with self.lock:
    if gpu_id in self.gpus:
        self.gpus[gpu_id].current_memory_usage = usage
```

锁粒度策略：
- 读取 `gpu_ids` 与回写共享状态时短持锁；NVML 采样等慢操作在锁外执行，降低阻塞。
- 使用 RLock 支持同线程嵌套获取，避免在方法套调用时死锁。

### 监控线程（monitor_thread）
`monitor_thread` 管理后台监控循环 `_monitor_loop` 的生命周期：启动时创建线程并置 `self.running=True`，停止时置 `self.running=False` 并 `join` 等待退出。

启动/停止：
```445:461:/workspace/server/nvgpu/gpu_manager.py
self.running = True
self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
self.monitor_thread.start()
# ...
self.running = False
if self.monitor_thread:
    self.monitor_thread.join(timeout=5)
```

监控循环职责：
```423:444:/workspace/server/nvgpu/gpu_manager.py
while self.running:
    # 读取 gpu_ids（锁内）→ 锁外逐张刷新内存 → 锁内检查严重错误暂停是否超时并自动恢复
    time.sleep(config.gpu_monitor_interval)
```

要点：
- 监控线程定期调用 `_update_gpu_memory` 刷新每张 GPU 的内存占用，并在需要时自动清除严重错误暂停状态。
- 与调度/管理接口共享同一把锁，确保采样写回与业务读写不产生竞态。

### 与 NVML 的交互
- 初始化：
```53:67:/workspace/server/nvgpu/gpu_manager.py
try:
    pynvml.nvmlInit()
    self.nvml_initialized = True
    device_count = pynvml.nvmlDeviceGetCount()
except Exception:
    self.nvml_initialized = False
```
- 采样：
```400:421:/workspace/server/nvgpu/gpu_manager.py
handle = pynvml.nvmlDeviceGetHandleByIndex(nvidia_smi_id)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
usage = mem_info.used / mem_info.total
with self.lock:
    # 回写 current_memory_usage
```

### 设计小结
- 使用可重入锁保护共享状态，分离“慢操作在锁外、状态写回在锁内”，兼顾一致性与吞吐。
- 监控线程只持有短锁窗口，避免长时间阻塞注册/调度等前台操作。
- 严重错误流程在锁内捕获快照、锁外处理杀进程与重入队，最终在锁内清理与状态复位。


