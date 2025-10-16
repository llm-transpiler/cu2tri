# NVGPU Server 核心模块函数详解

本文档详细解释核心模块中每个重要函数的作用、参数、前提条件和副作用。

**版本:** v2.5  
**最后更新:** 2025-01-15

---

## 目录

1. [数据模型 (models.py)](#数据模型-modelspy)
2. [GPU 管理器 (gpu_manager.py)](#gpu-管理器-gpu_managerpy)
3. [任务队列 (task_queue.py)](#任务队列-task_queuepy)
4. [任务执行器 (task_runner.py)](#任务执行器-task_runnerpy)
5. [调度器 (scheduler.py)](#调度器-schedulerpy)
6. [Python 客户端 (client.py)](#python-客户端-clientpy)

---

## 数据模型 (models.py)

### GPU 类

#### `can_accept_task(task_mode: TaskMode | None) -> bool`

**作用**: 判断 GPU 是否可以接受新任务

**参数**:
- `task_mode`: 任务模式（EXCLUSIVE/SHARED），如果为 None 则根据当前 GPU 模式判断

**返回**: True 表示可以接受，False 表示不能接受

**判断逻辑** (按优先级):

1. **GPU 状态检查**
   ```python
   if self.status != GPUStatus.ONLINE:
       return False
   ```
   - 前提: 无
   - 副作用: 无
   - 说明: 只有 ONLINE 状态的 GPU 才能接受任务

2. **获取有效模式**
   ```python
   effective_mode = self.manual_mode if self.manual_mode else self.mode
   ```
   - 说明: `manual_mode` (手动设置) 优先于 `mode` (任务驱动)
   - 这是 GPU 模式的核心决策逻辑

3. **独占任务检查** (`task_mode == TaskMode.EXCLUSIVE`)
   ```python
   return len(self.running_tasks) == 0
   ```
   - 要求: GPU 必须完全空闲
   - 不考虑内存和并发数限制
   - 即使 GPU 是 SHARED 模式也可以接受 EXCLUSIVE 任务（会切换模式）

4. **共享任务检查** (`task_mode == TaskMode.SHARED`)
   ```python
   # Step 1: 检查 GPU 是否在独占模式且有任务运行
   if effective_mode == GPUMode.EXCLUSIVE and len(self.running_tasks) > 0:
       return False
   
   # Step 2: 检查并发任务数限制
   if len(self.running_tasks) >= self.max_concurrent_tasks:
       return False
   
   # Step 3: 检查内存阈值
   return self.current_memory_usage < self.memory_threshold
   ```
   - **关键**: 独占模式且有任务 → 拒绝共享任务
   - 并发数达到上限 → 拒绝
   - 内存使用超过阈值 → 拒绝

**使用示例**:
```python
# 调度器检查 GPU 是否可接受任务
if gpu.can_accept_task(task.task_mode):
    # 可以分配任务
    scheduler.assign_task(task, gpu_id)
```

**特殊情况**:
- 独占模式 GPU 无任务时，可以接受共享任务（模式会切换）
- 共享任务可以与其他共享任务共存
- 独占任务必须等待 GPU 完全空闲

---

#### `is_available() -> bool`

**作用**: 简单检查 GPU 是否在线

**返回**: `status == GPUStatus.ONLINE`

**说明**: 
- 比 `can_accept_task()` 更简单的检查
- 不考虑任务数和内存
- 用于快速筛选可用 GPU

---

### Task 类

#### `to_dict() -> dict[str, Any]`

**作用**: 将 Task 对象转换为字典（用于 JSON 响应）

**返回**: 包含所有任务信息的字典

**关键逻辑**:
```python
# 1. 基本字段总是包含
result = {
    "task_id": self.task_id,
    "task_mode": self.task_mode.value,  # 转换为字符串
    "status": self.status.value,
    # ... 其他字段
}

# 2. 可选字段只在非 None 时包含
if self.task_type is not None:
    result["task_type"] = self.task_type.value

if self.task_label is not None:
    result["task_label"] = self.task_label
```

**设计原因**:
- 避免 JSON 中出现 `"task_type": null`
- 客户端可以通过 `"task_type" in data` 判断是否设置

---

## GPU 管理器 (gpu_manager.py)

### 核心管理函数

#### `register_gpu(gpu_id, mode=None, memory_threshold=None, max_concurrent_tasks=None) -> bool`

**作用**: 注册一个 GPU 供系统使用

**参数**:
- `gpu_id`: GPU 逻辑 ID
- `mode`: GPU 模式（默认使用 config.default_gpu_mode）
- `memory_threshold`: 内存阈值（默认使用 config.default_memory_threshold）
- `max_concurrent_tasks`: 最大并发任务数（默认使用 config.default_max_concurrent_tasks）

**前提条件**:
- GPU ID 未被注册

**副作用**:
- 在 `self.gpus` 字典中创建新的 GPU 对象
- 日志记录注册信息

**返回**: 
- `True`: 注册成功
- `False`: GPU ID 已存在

**代码逻辑**:
```python
def register_gpu(self, gpu_id: int, ...) -> bool:
    with self.lock:  # 线程安全
        # 1. 检查是否已注册
        if gpu_id in self.gpus:
            logger.warning(f"GPU {gpu_id} already registered")
            return False
        
        # 2. 创建 GPU 对象（应用默认值）
        gpu = GPU(
            gpu_id=gpu_id,
            mode=mode or config.default_gpu_mode,
            memory_threshold=memory_threshold or config.default_memory_threshold,
            max_concurrent_tasks=max_concurrent_tasks or config.default_max_concurrent_tasks,
            status=GPUStatus.ONLINE  # 默认在线
        )
        
        # 3. 添加到字典
        self.gpus[gpu_id] = gpu
        
        # 4. 记录日志
        logger.info(f"Registered GPU {gpu_id} with mode={gpu.mode}, ...")
        
        return True
```

---

#### `find_available_gpu(preferred_gpu=None, task_mode=None) -> int | None`

**作用**: 为任务找到可用的 GPU

**参数**:
- `preferred_gpu`: 优先使用的 GPU ID（用户指定）
- `task_mode`: 任务模式（EXCLUSIVE/SHARED）

**返回**:
- GPU ID: 找到可用 GPU
- `None`: 没有可用 GPU

**前提条件**:
- 已有注册的 GPU

**副作用**:
- **关键**: 更新 GPU 内存使用情况（调用 `_update_gpu_memory`）
- 无其他副作用

**逻辑详解**:

```python
def find_available_gpu(self, preferred_gpu=None, task_mode=None) -> int | None:
    # Phase 1: 严重错误检查（锁外，避免死锁）
    if self.severe_error_active:
        return None  # 系统暂停，不分配任务
    
    # Phase 2: 优先 GPU 检查
    if preferred_gpu is not None and preferred_gpu in self.gpus:
        # 更新内存（关键！确保最新数据）
        self._update_gpu_memory(preferred_gpu)
        
        with self.lock:
            gpu = self.gpus.get(preferred_gpu)
            if gpu and gpu.can_accept_task(task_mode):
                return preferred_gpu
        return None  # 优先 GPU 不可用时，不尝试其他 GPU
    
    # Phase 3: 查找任何可用 GPU
    # Step 1: 更新所有 GPU 内存（锁外，避免死锁）
    with self.lock:
        gpu_ids = list(self.gpus.keys())
    
    for gpu_id in gpu_ids:
        self._update_gpu_memory(gpu_id)  # 更新每个 GPU 的内存
    
    # Step 2: 检查哪个 GPU 可用
    with self.lock:
        for gpu_id, gpu in self.gpus.items():
            if gpu.can_accept_task(task_mode):
                return gpu_id  # 返回第一个可用的
    
    return None  # 没有可用 GPU
```

**关键设计点**:

1. **内存实时更新**
   - 在检查前主动更新 GPU 内存
   - 解决任务延迟分配内存的问题
   - 调度决策基于最新数据

2. **锁的使用**
   - `_update_gpu_memory` 在锁外调用（内部有自己的锁）
   - 避免死锁（NVML 调用可能耗时）

3. **优先 GPU 逻辑**
   - 用户指定 GPU 时，只检查该 GPU
   - 不可用时返回 None（不尝试其他 GPU）

---

#### `set_gpu_mode_for_task(gpu_id, task, task_mode) -> bool`

**作用**: 根据任务需求设置 GPU 模式（任务驱动模式切换）

**调用时机**: 任务启动时（在 `scheduler._schedule_round()` 中）

**参数**:
- `gpu_id`: GPU ID
- `task_id`: 任务 ID
- `task_mode`: 任务模式（EXCLUSIVE/SHARED）

**前提条件**:
- GPU 已注册
- 任务已被分配到该 GPU

**副作用**:
- 可能改变 GPU 的 `mode` 字段
- 独占任务会设置 `mode_locked_by` 字段
- 记录日志

**逻辑详解**:

```python
def set_gpu_mode_for_task(self, gpu_id, task, task_mode):
    with self.lock:
        gpu = self.gpus[gpu_id]
        
        # 1. 尊重手动模式
        if gpu.manual_mode:
            logger.debug(f"GPU {gpu_id} has manual mode, not changing")
            return True  # 不改变模式，但返回成功
        
        # 2. 独占任务 → 设置为独占模式
        if task_mode == TaskMode.EXCLUSIVE:
            gpu.mode = GPUMode.EXCLUSIVE
            gpu.mode_locked_by = task_id  # 锁定模式
            logger.info(f"GPU {gpu_id} mode set to EXCLUSIVE for task {task_id[:8]}")
        
        # 3. 共享任务 → 只在未锁定时设置为共享模式
        else:  # SHARED
            if not gpu.mode_locked_by:
                gpu.mode = GPUMode.SHARED
                logger.debug(f"GPU {gpu_id} mode set to SHARED")
        
        return True
```

**关键逻辑**:

1. **手动模式优先**
   - 管理员设置的手动模式不被任务覆盖
   - 保护管理员意图

2. **独占任务锁定**
   - 独占任务运行时锁定 GPU 模式
   - 其他任务无法改变模式

3. **共享任务不强制**
   - 如果 GPU 被独占任务锁定，共享任务不改变模式
   - 避免冲突

---

#### `restore_gpu_mode_after_task(gpu_id, task_id) -> bool`

**作用**: 任务完成后恢复 GPU 模式

**调用时机**: 任务结束时（在 `scheduler._execute_task()` 的 finally 块中）

**参数**:
- `gpu_id`: GPU ID
- `task_id`: 刚完成的任务 ID

**前提条件**:
- GPU 已注册
- 任务已经或正在完成

**副作用**:
- 可能清除 `mode_locked_by`
- 可能恢复 GPU 模式为 `manual_mode` 或 SHARED
- 记录日志

**逻辑详解**:

```python
def restore_gpu_mode_after_task(self, gpu_id, task_id):
    with self.lock:
        gpu = self.gpus[gpu_id]
        
        # 1. 检查是否是锁定此模式的任务
        if gpu.mode_locked_by == task_id:
            # 解锁
            gpu.mode_locked_by = None
            logger.debug(f"GPU {gpu_id} mode unlocked by task {task_id[:8]}")
            
            # 2. 恢复模式
            if gpu.manual_mode:
                # 恢复到手动模式
                gpu.mode = gpu.manual_mode
                logger.debug(f"GPU {gpu_id} mode restored to manual mode")
            else:
                # 恢复到默认共享模式（如果无任务运行）
                if len(gpu.running_tasks) == 0:
                    gpu.mode = GPUMode.SHARED
                    logger.debug(f"GPU {gpu_id} mode restored to SHARED")
        
        return True
```

**关键逻辑**:

1. **只解锁自己锁定的模式**
   - 检查 `mode_locked_by == task_id`
   - 避免误解锁

2. **恢复优先级**
   - 手动模式 > 默认共享模式
   - 保护管理员意图

3. **条件恢复**
   - 只在无运行任务时恢复为 SHARED
   - 避免与其他任务冲突

---

#### `trigger_severe_error(gpu_id, error_msg)`

**作用**: 触发严重错误处理流程

**调用时机**:
- GPU 硬件错误
- CUDA 错误
- 测试用（API 触发）

**参数**:
- `gpu_id`: 出错的 GPU ID
- `error_msg`: 错误消息

**前提条件**:
- 已设置依赖项（`set_dependencies` 调用过）

**副作用** (非常重要):
1. 设置全局 `severe_error_active = True`
2. 记录错误时间 `severe_error_time = datetime.now()`
3. GPU 状态设为 ERROR
4. **强制终止所有运行中的任务** (调用 `task_runner.kill_task`)
5. **将被终止的任务重新排队到队首** (调用 `task_queue.push_front`)
6. 清空 GPU 的 `running_tasks` 列表
7. 记录错误日志
8. **暂停调度 60 秒**（自动恢复）

**逻辑详解**:

```python
def trigger_severe_error(self, gpu_id, error_msg):
    # Phase 1: 更新状态（锁内）
    with self.lock:
        self.severe_error_active = True
        self.severe_error_time = datetime.now()
        
        if gpu_id in self.gpus:
            self.gpus[gpu_id].status = GPUStatus.ERROR
            self.gpus[gpu_id].error_message = error_msg
            self.gpus[gpu_id].last_error_time = self.severe_error_time
            running_task_ids = list(self.gpus[gpu_id].running_tasks)  # 复制列表
        else:
            running_task_ids = []
        
        logger.error(f"SEVERE ERROR on GPU {gpu_id}: {error_msg}")
    
    # Phase 2: 终止和重新排队任务（锁外，避免死锁）
    if self.task_queue and self.task_runner and running_task_ids:
        killed_tasks = []
        
        # Step 1: 终止所有运行任务
        for task_id in running_task_ids:
            task = self.task_queue.get_task(task_id)
            if task:
                if self.task_runner.kill_task(task_id):
                    logger.info(f"Killed task {task_id[:8]}")
                    killed_tasks.append(task)
        
        # Step 2: 重新排队（逆序，保持原顺序）
        for task in reversed(killed_tasks):
            # 重置任务状态
            task.start_timestamp = None
            task.end_timestamp = None
            task.exit_code = None
            task.error_message = f"Requeued due to GPU {gpu_id} severe error"
            
            # 插入队首
            self.task_queue.push_front(task)
            logger.info(f"Requeued task {task.task_id[:8]}")
        
        # Step 3: 清空 GPU 任务列表
        with self.lock:
            if gpu_id in self.gpus:
                self.gpus[gpu_id].running_tasks.clear()
    
    logger.error(f"SEVERE ERROR handling complete. System paused for {config.error_pause_duration}s.")
```

**设计考虑**:

1. **分阶段执行**
   - 先更新状态（快速，锁内）
   - 再处理任务（慢速，锁外）
   - 避免长时间持锁

2. **任务保护**
   - 被终止的任务不丢失
   - 重新排队到队首（高优先级）
   - 保留任务 ID 和配置

3. **自动恢复**
   - 监控线程检测超时
   - 60 秒后自动清除错误状态
   - 无需人工干预

---

## 任务队列 (task_queue.py)

### 核心队列函数

#### `submit_task(task: Task) -> str`

**作用**: 提交新任务到全局队列

**参数**:
- `task`: Task 对象

**前提条件**:
- Task 对象已创建
- `task_id` 已生成

**副作用**:
- 设置任务状态为 PENDING
- 添加到 `self.tasks` 字典
- 添加到 `self.global_queue` 队尾
- 记录日志

**返回**: 任务 ID

**代码逻辑**:
```python
def submit_task(self, task: Task) -> str:
    with self.lock:
        # 1. 设置状态
        task.status = TaskStatus.PENDING
        
        # 2. 添加到字典（用于查询）
        self.tasks[task.task_id] = task
        
        # 3. 添加到全局队列（FIFO）
        self.global_queue.append(task)
    
    # 4. 记录日志（锁外）
    logger.info(f"Task {task.task_id} submitted: mode={task.task_mode.value}, ...")
    
    return task.task_id
```

---

#### `queue_task_for_gpu(task: Task, gpu_id: int)`

**作用**: 将任务分配到特定 GPU 的队列

**调用时机**: 调度器找到可用 GPU 后

**参数**:
- `task`: Task 对象
- `gpu_id`: 目标 GPU ID

**前提条件**:
- 任务状态为 PENDING
- GPU 已检查可接受此任务

**副作用**:
- 任务状态改为 QUEUED
- 设置 `task.assigned_gpu = gpu_id`
- 添加到 `self.gpu_queues[gpu_id]`
- 如果 GPU 队列不存在则创建
- 记录日志

**代码逻辑**:
```python
def queue_task_for_gpu(self, task: Task, gpu_id: int):
    with self.lock:
        # 1. 确保 GPU 队列存在
        if gpu_id not in self.gpu_queues:
            self.gpu_queues[gpu_id] = deque()
        
        # 2. 更新任务状态
        task.status = TaskStatus.QUEUED
        task.assigned_gpu = gpu_id
        
        # 3. 添加到 GPU 队列
        self.gpu_queues[gpu_id].append(task)
        
        # 4. 记录日志
    logger.info(f"Task {format_task_ref(task)} queued for GPU {gpu_id}")
```

**状态转换**: `PENDING → QUEUED`

---

#### `cancel_task(task_id: str) -> bool`

**作用**: 取消 PENDING 或 QUEUED 状态的任务

**参数**:
- `task_id`: 任务 ID

**前提条件**:
- 任务存在
- 任务状态为 PENDING 或 QUEUED

**副作用**:
- 从全局队列或 GPU 队列中移除任务
- 设置任务状态为 CANCELLED
- 记录日志

**返回**:
- `True`: 取消成功
- `False`: 任务不存在或状态不允许取消

**代码逻辑**:
```python
def cancel_task(self, task_id: str) -> bool:
    with self.lock:
        # 1. 获取任务
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # 2. 检查状态
        if task.status not in [TaskStatus.PENDING, TaskStatus.QUEUED]:
            logger.warning(f"Cannot cancel task {task_id} with status {task.status.value}")
            return False
        
        # 3. 从全局队列移除
        try:
            self.global_queue.remove(task)
        except ValueError:
            pass  # 可能已不在全局队列
        
        # 4. 从 GPU 队列移除
        if task.assigned_gpu is not None:
            gpu_id = task.assigned_gpu
            if gpu_id in self.gpu_queues:
                try:
                    self.gpu_queues[gpu_id].remove(task)
                except ValueError:
                    pass
        
        # 5. 设置状态
        task.status = TaskStatus.CANCELLED
        logger.info(f"Task {task_id} cancelled")
        
        return True
```

**注意**: 
- 运行中的任务不能用此方法取消
- 需要使用 `force_cancel_task`

---

#### `force_cancel_task(task_id: str, task_runner) -> bool`

**作用**: 强制取消任务，包括运行中的任务

**参数**:
- `task_id`: 任务 ID
- `task_runner`: TaskRunner 实例（用于终止进程）

**前提条件**:
- 任务存在
- task_runner 已初始化

**副作用**:
- 对于 PENDING/QUEUED: 调用 `cancel_task`
- 对于 RUNNING: 
  - 调用 `task_runner.kill_task` 终止进程
  - 设置状态为 CANCELLED
  - 设置 `error_message` 和 `end_timestamp`
- 记录日志

**返回**:
- `True`: 取消成功
- `False`: 任务不存在、已完成或终止失败

**代码逻辑**:
```python
def force_cancel_task(self, task_id: str, task_runner) -> bool:
    with self.lock:
        if task_id not in self.tasks:
            logger.warning(f"Cannot force cancel task {task_id}: not found")
            return False
        
        task = self.tasks[task_id]
        
        # Case 1: PENDING/QUEUED - 使用普通取消
        if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
            return self.cancel_task(task_id)
        
        # Case 2: RUNNING - 强制终止
        if task.status == TaskStatus.RUNNING:
            logger.info(f"Force cancelling running task {task_id}")
            
            # 终止进程（锁内调用，task_runner 有自己的锁）
            if task_runner.kill_task(task_id):
                # 更新任务状态
                task.status = TaskStatus.CANCELLED
                task.error_message = "Cancelled by user (force)"
                task.end_timestamp = datetime.now()
                logger.info(f"Task {task_id} force cancelled")
                return True
            else:
                logger.error(f"Failed to kill running task {task_id}")
                return False
        
        # Case 3: 已完成的任务
        logger.warning(f"Cannot cancel task {task_id} with status {task.status.value}")
        return False
```

**关键点**:
- 运行中的任务需要终止进程
- 失败时返回 False
- 已完成的任务不能取消

---

## 任务执行器 (task_runner.py)

### 核心执行函数

#### `run_task(task: Task, gpu_id: int) -> bool`

**作用**: 在指定 GPU 上执行任务

**参数**:
- `task`: Task 对象
- `gpu_id`: 目标 GPU ID

**前提条件**:
- 任务已分配到 GPU
- GPU 可用

**副作用**:
- 更新任务状态: QUEUED → RUNNING → COMPLETED/FAILED/CANCELLED
- 设置 `task.assigned_gpu`, `start_timestamp`, `end_timestamp`, `exit_code`
- 创建 stdout/stderr 文件
- 创建日志文件
- 注册/注销运行进程
- 记录详细日志

**返回**:
- `True`: 任务启动成功（不论执行结果）
- `False`: 任务启动失败（异常）

**代码流程** (详细):

```python
def run_task(self, task: Task, gpu_id: int) -> bool:
    stdout_file = None
    stderr_file = None
    
    try:
        # === Phase 1: 准备阶段 ===
        
        # 1.1 更新任务状态
        task.status = TaskStatus.RUNNING
        task.assigned_gpu = gpu_id
        task.start_timestamp = datetime.now()
        
        # 1.2 准备脚本路径
        script_abs_path = os.path.abspath(task.script_path)
        logger.info(f"Starting task {task.task_id} on GPU {gpu_id}")
        
        # 1.3 准备环境变量
        env = os.environ.copy()
        
        # 映射 GPU ID 到 CUDA_VISIBLE_DEVICES
        cuda_id = gpu_id
        if gpu_config_loader:
            cuda_visible_id = gpu_config_loader.get_cuda_visible_id(gpu_id)
            if cuda_visible_id is not None:
                cuda_id = cuda_visible_id
        
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
        
        # 合并自定义环境变量
        if task.env:
            env.update(task.env)
        
        # 1.4 准备日志目录
        log_dir = NVGPU_ROOT / "logs" / "tasks"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 1.5 设置日志文件路径
        task.log_file = str(log_dir / f"{task.task_id}.log")
        stdout_path = log_dir / f"{task.task_id}.stdout"
        stderr_path = log_dir / f"{task.task_id}.stderr"
        
        # 1.6 准备命令
        cmd = [sys.executable, script_abs_path] + task.args
        logger.debug(f"Task {task.task_id} command: {' '.join(cmd)}")
        
        # === Phase 2: 执行阶段 ===
        
        # 2.1 打开输出文件（行缓冲）
        stdout_file = open(stdout_path, 'w', buffering=1)
        stderr_file = open(stderr_path, 'w', buffering=1)
        
        # 2.2 启动子进程
        process = subprocess.Popen(
            cmd,
            cwd=task.work_dir,  # 工作目录
            env=env,            # 环境变量
            stdout=stdout_file,  # 重定向 stdout
            stderr=stderr_file   # 重定向 stderr
        )
        
        # 2.3 注册进程（用于 kill_task）
        with self.lock:
            self.running_processes[task.task_id] = process
        
        try:
            # 2.4 等待进程完成（带超时）
            exit_code = process.wait(timeout=config.task_timeout)
        except subprocess.TimeoutExpired:
            # 超时处理
            logger.warning(f"Task {task.task_id} timed out, terminating...")
            process.kill()
            process.wait()
            raise  # 重新抛出，由外层 except 处理
        finally:
            # 2.5 注销进程
            with self.lock:
                self.running_processes.pop(task.task_id, None)
        
        # === Phase 3: 结果处理 ===
        
        # 3.1 关闭文件并获取大小
        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None
        
        task.stdout_size = stdout_path.stat().st_size
        task.stderr_size = stderr_path.stat().st_size
        
        # 3.2 保存退出码和结束时间
        task.exit_code = exit_code
        task.end_timestamp = datetime.now()
        
        # 3.3 确定最终状态
        if exit_code == 0:
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task {task.task_id} completed successfully")
        elif exit_code < 0:
            # 信号终止（如 SIGSEGV）
            signal_name = self._get_signal_name(abs(exit_code))
            # 不覆盖 CANCELLED 状态
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.FAILED
                task.error_message = f"Process terminated by signal {signal_name}"
            logger.error(f"Task {task.task_id} terminated by signal {signal_name}")
        else:
            # 非零退出码
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.FAILED
                task.error_message = f"Exit code {exit_code}"
            logger.warning(f"Task {task.task_id} failed with exit code {exit_code}")
        
        # 3.4 写入汇总日志
        self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path)
        
        return True
    
    except subprocess.TimeoutExpired:
        # 超时异常处理
        task.status = TaskStatus.FAILED
        task.error_message = f"Timeout after {config.task_timeout} seconds"
        task.end_timestamp = datetime.now()
        
        # 关闭文件
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
        
        # 获取文件大小
        try:
            if stdout_path.exists():
                task.stdout_size = stdout_path.stat().st_size
            if stderr_path.exists():
                task.stderr_size = stderr_path.stat().st_size
        except:
            pass
        
        self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path, timeout=True)
        return True
    
    except Exception as e:
        # 其他异常处理
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.end_timestamp = datetime.now()
        logger.error(f"Task {task.task_id} failed with exception: {e}")
        
        # 关闭文件
        if stdout_file:
            stdout_file.close()
        if stderr_file:
            stderr_file.close()
        
        # 获取文件大小
        try:
            if stdout_path and stdout_path.exists():
                task.stdout_size = stdout_path.stat().st_size
            if stderr_path and stderr_path.exists():
                task.stderr_size = stderr_path.stat().st_size
        except:
            pass
        
        self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path, exception=str(e))
        return False
```

**关键设计点**:

1. **状态管理**
   - 先设置 RUNNING，再启动进程
   - 根据 exit_code 确定最终状态
   - 不覆盖 CANCELLED 状态

2. **资源管理**
   - 使用 try-finally 确保资源清理
   - 注册/注销进程
   - 关闭文件句柄

3. **日志处理**
   - stdout/stderr 分别存储
   - 行缓冲模式（实时写入）
   - 记录文件大小

4. **错误处理**
   - 超时: FAILED + 特定错误信息
   - 信号终止: 识别信号类型
   - 异常: 记录异常信息

---

#### `kill_task(task_id: str) -> bool`

**作用**: 强制终止运行中的任务

**参数**:
- `task_id`: 任务 ID

**前提条件**:
- 任务正在运行
- 进程已注册

**副作用**:
- 终止子进程（SIGTERM 或 SIGKILL）
- 不改变任务状态（由调用者负责）
- 记录日志

**返回**:
- `True`: 终止成功
- `False`: 任务不在运行或终止失败

**代码逻辑**:
```python
def kill_task(self, task_id: str) -> bool:
    with self.lock:
        # 1. 获取进程
        process = self.running_processes.get(task_id)
        if not process:
            logger.warning(f"Cannot kill task {task_id}: not running")
            return False
        
        try:
            # 2. 优雅终止（SIGTERM）
            logger.info(f"Terminating task {task_id}...")
            process.terminate()
            
            # 3. 等待 5 秒
            try:
                process.wait(timeout=5)
                logger.info(f"Task {task_id} terminated gracefully")
            except subprocess.TimeoutExpired:
                # 4. 强制终止（SIGKILL）
                logger.warning(f"Task {task_id} did not terminate, sending SIGKILL...")
                process.kill()
                process.wait()
                logger.info(f"Task {task_id} killed forcefully")
            
            return True
        except Exception as e:
            logger.error(f"Failed to kill task {task_id}: {e}")
            return False
```

**终止策略**:
1. 先尝试优雅终止 (SIGTERM)
2. 等待 5 秒
3. 超时后强制终止 (SIGKILL)

---

## 调度器 (scheduler.py)

### 核心调度函数

#### `_schedule_round()`

**作用**: 执行一轮完整的任务调度

**调用时机**: 每秒一次（`scheduler_interval`）

**前提条件**:
- 调度器已启动
- 无严重错误（或已恢复）

**副作用**:
- 将 PENDING 任务分配到 GPU 队列（状态变为 QUEUED）
- 启动 QUEUED 任务（状态变为 RUNNING）
- 创建执行线程
- 更新 GPU 状态
- 记录日志

**两阶段调度**:

```python
def _schedule_round(self):
    # === Phase 1: 分配待处理任务到 GPU 队列 ===
    while True:
        # 1.1 从全局队列弹出任务
        task = self.task_queue.pop_pending_task()
        if not task:
            break  # 队列空
        
        # 1.2 查找可用 GPU
        gpu_id = self.gpu_manager.find_available_gpu(
            task.gpu_id,      # 优先 GPU
            task.task_mode    # 任务模式
        )
        
        if gpu_id is None:
            # 没有可用 GPU，放回队列
            self.task_queue.global_queue.appendleft(task)
            break  # 停止分配，等待下一轮
        
        # 1.3 分配到 GPU 队列
        self.task_queue.queue_task_for_gpu(task, gpu_id)
    
    # === Phase 2: 执行 GPU 队列中的任务 ===
    for gpu in self.gpu_manager.list_gpus():
        gpu_id = gpu.gpu_id
        
        # 2.1 从 GPU 队列弹出任务
        task = self.task_queue.pop_gpu_task(gpu_id)
        if not task:
            continue  # 该 GPU 无任务
        
        # 2.2 再次检查 GPU 是否可接受（可能状态已变）
        if not gpu.can_accept_task(task.task_mode):
            # 放回 GPU 队列
            self.task_queue.gpu_queues[gpu_id].appendleft(task)
            continue
        
        # 2.3 标记任务为运行（关键：在主线程中）
        self.gpu_manager.mark_task_running(gpu_id, task)
        
        # 2.4 设置 GPU 模式（关键：在主线程中）
        self.gpu_manager.set_gpu_mode_for_task(gpu_id, task, task.task_mode)
        
        # 2.5 创建执行线程
        thread = threading.Thread(
            target=self._execute_task,
            args=(task, gpu_id),
            daemon=True
        )
        thread.start()
```

**关键设计点**:

1. **两阶段分离**
   - Phase 1: 全局队列 → GPU 队列
   - Phase 2: GPU 队列 → 执行
   - 好处：清晰的职责分离

2. **Race Condition 防护**
   - 在主线程标记 RUNNING
   - 在主线程设置 GPU 模式
   - **然后**启动执行线程
   - 保证状态一致性

3. **回退机制**
   - Phase 1: 无可用 GPU → 放回全局队列
   - Phase 2: GPU 状态变化 → 放回 GPU 队列

4. **FIFO 保证**
   - 使用 `deque.popleft()` 和 `appendleft()`
   - 保持任务顺序

---

#### `_execute_task(task, gpu_id)`

**作用**: 在独立线程中执行单个任务

**调用时机**: 由 `_schedule_round()` 创建线程调用

**参数**:
- `task`: Task 对象
- `gpu_id`: 分配的 GPU ID

**前提条件**:
- 任务已标记为 RUNNING（在主线程）
- GPU 模式已设置（在主线程）

**副作用**:
- 执行任务（调用 `task_runner.run_task`）
- 检查 GPU 错误（可选）
- 恢复 GPU 模式
- 标记任务完成
- 记录日志

**代码逻辑**:
```python
def _execute_task(self, task, gpu_id: int):
    """注意：任务已在主线程中标记为 RUNNING，GPU 模式已设置"""
    
    try:
        # 1. 执行任务
        success = self.task_runner.run_task(task, gpu_id)
        
        # 2. 可选：检查 GPU 错误
        if not success or task.exit_code != 0:
            # 检查 stderr 中的 GPU 错误
            if task.log_file and task.stderr_size > 0 and task.stderr_size < 1MB:
                try:
                    stderr_path = Path(task.log_file).parent / f"{task.task_id}.stderr"
                    if stderr_path.exists():
                        stderr_content = stderr_path.read_text(...)
                        # 检查 GPU 相关错误
                        if any(err in stderr_content.lower() 
                               for err in ["cuda error", "gpu error", "out of memory"]):
                            logger.warning(f"GPU error detected in task {task.task_id}")
                            # 可选：触发 severe error
                            # self.gpu_manager.trigger_severe_error(gpu_id, "GPU error")
                except Exception as e:
                    logger.debug(f"Could not check stderr for GPU errors: {e}")
    
    finally:
        # 3. 恢复 GPU 模式（总是执行）
        self.gpu_manager.restore_gpu_mode_after_task(gpu_id, task.task_id)

        # 4. 标记任务完成（总是执行）
        self.gpu_manager.mark_task_completed(gpu_id, task)
```

**关键点**:

1. **finally 块**
   - 无论任务成功失败都执行
   - 确保资源清理
   - 恢复 GPU 状态

2. **GPU 错误检测**
   - 可选功能（目前未启用）
   - 检查 stderr 中的错误关键字
   - 可触发 severe error

3. **线程独立**
   - 每个任务独立线程
   - 互不影响
   - daemon 线程（主程序退出时自动终止）

---

## Python 客户端 (client.py)

### 核心客户端函数

#### `submit_task(...) -> str`

**作用**: 提交任务到服务器

**参数** (简化版):
- `script_path`: 脚本路径
- `task_mode`: 任务模式（可选，智能默认）
- `task_type`: 任务类型（可选）
- `task_label`: 任务标签（可选）
- `work_dir`: 工作目录
- `args`: 命令行参数
- `env`: 环境变量
- `gpu_id`: 指定 GPU

**前提条件**:
- 服务器运行中
- 脚本路径有效

**副作用**:
- 发送 HTTP POST 请求到 `/tasks`
- 服务器创建任务并排队

**返回**: 任务 ID

**代码逻辑**:
```python
def submit_task(self, script_path, ...) -> str:
    # 1. 构建请求体（只包含非 None 字段）
    payload = {
        "script_path": script_path,
        "work_dir": work_dir,
        "args": args or [],
    }
    
    # 2. 添加可选字段
    if task_mode is not None:
        payload["task_mode"] = task_mode
    if task_type is not None:
        payload["task_type"] = task_type
    if task_label is not None:
        payload["task_label"] = task_label
    if env:
        payload["env"] = env
    if gpu_id is not None:
        payload["gpu_id"] = gpu_id
    
    # 3. 发送请求
    response = self.session.post(
        f"{self.base_url}/tasks",
        json=payload,
        timeout=10
    )
    
    # 4. 检查响应
    if response.status_code != 200:
        raise RuntimeError(f"Failed to submit task: {response.text}")
    
    # 5. 返回任务 ID
    result = response.json()
    return result["task_id"]
```

**智能默认值**:
- 在服务器端应用（`api_server.py`）
- 客户端不需要处理逻辑

---

#### `wait_for_task(task_id, timeout=None, poll_interval=2.0) -> TaskResult`

**作用**: 等待任务完成

**参数**:
- `task_id`: 任务 ID
- `timeout`: 超时时间（秒），None 表示无限等待
- `poll_interval`: 轮询间隔（秒）

**前提条件**:
- 任务已提交

**副作用**:
- 周期性调用 `get_task`
- 消耗网络带宽

**返回**: TaskResult 对象

**抛出**:
- `TimeoutError`: 超时
- `RuntimeError`: 任务不存在

**代码逻辑**:
```python
def wait_for_task(self, task_id, timeout=None, poll_interval=2.0) -> TaskResult:
    start_timestamp = time.time()
    
    while True:
        # 1. 获取任务状态
        result = self.get_task(task_id)
        
        # 2. 检查是否完成
        if result.status in ["completed", "failed", "cancelled"]:
            return result
        
        # 3. 检查超时
        if timeout and (time.time() - start_timestamp) > timeout:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
        
        # 4. 等待下次轮询
        time.sleep(poll_interval)
```

**使用模式**:
```python
# 提交任务
task_id = client.submit_task("test.py")

# 等待完成
try:
    result = client.wait_for_task(task_id, timeout=300)
    if result.status == "completed":
        print("Success!")
except TimeoutError:
    client.cancel_task(task_id, force=True)
```

---

#### `cancel_task(task_id, force=False) -> bool`

**作用**: 取消任务

**参数**:
- `task_id`: 任务 ID
- `force`: 是否强制取消（终止运行中的任务）

**前提条件**:
- 任务存在

**副作用**:
- 发送 HTTP POST 请求到 `/tasks/{task_id}/cancel`
- 服务器取消任务

**返回**: True 表示成功

**抛出**: `RuntimeError` 如果取消失败

**代码逻辑**:
```python
def cancel_task(self, task_id, force=False) -> bool:
    # 发送取消请求
    response = self.session.post(
        f"{self.base_url}/tasks/{task_id}/cancel",
        params={"force": force},
        timeout=10
    )
    
    # 检查响应
    if response.status_code != 200:
        raise RuntimeError(f"Failed to cancel task: {response.text}")
    
    return True
```

**使用场景**:
```python
# 取消排队任务
client.cancel_task(task_id)  # force=False (默认)

# 强制取消运行任务
client.cancel_task(task_id, force=True)
```

---

## 总结

### 线程安全保证

所有核心模块使用 `threading.RLock()` 保护共享状态：

1. **GPUManager**: 保护 GPU 字典和状态
2. **TaskQueue**: 保护任务队列和字典
3. **TaskRunner**: 保护运行进程字典

### 关键设计模式

1. **两阶段提交**
   - 调度器：全局队列 → GPU 队列 → 执行
   - 保证原子性和一致性

2. **Race Condition 防护**
   - 先更新状态，再启动线程
   - 在主线程完成关键操作

3. **错误隔离**
   - 任务级错误不影响其他任务
   - GPU 级错误可恢复
   - 系统级错误有自动恢复

4. **资源管理**
   - 使用 try-finally 确保清理
   - 分离锁内外操作
   - 避免死锁

### 常见陷阱

1. **不要在锁内做耗时操作**
   - NVML 调用在锁外
   - 文件 I/O 在锁外
   - 进程管理在锁外

2. **注意状态一致性**
   - 任务状态与队列位置
   - GPU 模式与运行任务
   - 运行进程与任务状态

3. **锁的获取顺序**
   - 避免嵌套锁
   - 使用 RLock 允许重入

---

## 相关文档

- [配置模块详解](CODE_REFERENCE_CONFIG.md) - 枚举和配置
- [API 参考手册](API_REFERENCE_ZH.md) - REST API 文档
- [设计文档](DESIGN.md) - 系统设计理念
- [测试覆盖](../tests/TEST_COVERAGE.md) - 测试说明

---

**维护者**: 修改代码后请同步更新此文档
