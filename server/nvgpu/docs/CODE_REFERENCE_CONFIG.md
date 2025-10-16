# 配置模块详解 (config.py)

本文档详细解释 `config.py` 模块的实现。

**文件路径:** `server/nvgpu/config.py`  
**依赖:** 无（基础模块）  
**被依赖:** 所有其他模块

---

## 模块概述

`config.py` 定义了 NVGPU Server 的所有枚举类型和配置参数。这是系统的基础模块，不依赖任何其他模块。

### 主要内容

1. **枚举定义** (5 个)
   - GPUMode - GPU 执行模式
   - TaskType - 任务业务类型
   - TaskMode - 任务执行模式
   - TaskStatus - 任务状态
   - GPUStatus - GPU 状态

2. **配置类** (1 个)
   - ServerConfig - 服务器配置参数

---

## 枚举定义

### 1. GPUMode - GPU 执行模式

```python
class GPUMode(str, Enum):
    """GPU execution mode."""
    EXCLUSIVE = "exclusive"  # Only one task at a time (exclusive access)
    SHARED = "shared"        # Multiple tasks allowed (shared access)
```

**用途**: 控制 GPU 如何执行任务

**取值**:
- `EXCLUSIVE`: 独占模式，同时只允许一个任务运行
- `SHARED`: 共享模式，允许多个任务同时运行

**设计说明**:
- 继承 `str` 和 `Enum`，可直接用作字符串
- GPU 的 `mode` 字段使用此枚举
- 可以被 `manual_mode` 覆盖（管理员设置）
- 可以被 `mode_locked_by` 锁定（独占任务运行时）

**使用场景**:
```python
# 设置 GPU 为独占模式
gpu.mode = GPUMode.EXCLUSIVE

# 比较模式
if gpu.effective_mode == GPUMode.EXCLUSIVE:
    # ...

# 作为字符串使用
print(f"Mode: {gpu.mode.value}")  # "exclusive" 或 "shared"
```

---

### 2. TaskType - 任务业务类型

```python
class TaskType(str, Enum):
    """Task type for business categorization."""
    FUNCTIONAL = "functional"  # Functional test
    PERFORMANCE = "performance"  # Performance test
    BOTH = "both"  # Both functional and performance
```

**用途**: 任务的业务层面分类，用于统计、筛选和智能默认值

**取值**:
- `FUNCTIONAL`: 功能测试（一般可以共享 GPU）
- `PERFORMANCE`: 性能测试（一般需要独占 GPU）
- `BOTH`: 既有功能测试又有性能测试

**智能默认值逻辑**:
```python
# 在 models.py Task.__post_init__() 中实现
if task_type == TaskType.FUNCTIONAL:
    # 默认使用 shared 模式
    task_mode = TaskMode.SHARED
elif task_type == TaskType.PERFORMANCE or task_type == TaskType.BOTH:
    # 默认使用 exclusive 模式
    task_mode = TaskMode.EXCLUSIVE
```

**可选性**: Task.task_type 可以为 None（用户不指定类型）

**使用场景**:
```python
# 提交功能测试（自动 shared）
task = Task(
    script_path="test.py",
    task_type=TaskType.FUNCTIONAL  # 自动设置 task_mode=SHARED
)

# 提交性能测试（自动 exclusive）
task = Task(
    script_path="benchmark.py",
    task_type=TaskType.PERFORMANCE  # 自动设置 task_mode=EXCLUSIVE
)

# 筛选任务
functional_tasks = [t for t in tasks if t.task_type == TaskType.FUNCTIONAL]
```

---

### 3. TaskMode - 任务执行模式

```python
class TaskMode(str, Enum):
    """Task execution mode (controls GPU behavior)."""
    EXCLUSIVE = "exclusive"  # Task requires exclusive GPU access
    SHARED = "shared"        # Task can share GPU with others
```

**用途**: 控制任务如何使用 GPU（是否可与其他任务共享）

**取值**:
- `EXCLUSIVE`: 任务需要独占 GPU
- `SHARED`: 任务可以与其他任务共享 GPU

**与 GPUMode 的区别**:
- `TaskMode`: 任务属性，表达任务的需求
- `GPUMode`: GPU 属性，表示 GPU 当前的运行模式

**关系**:
```
Task.task_mode (任务需求) 
    → 影响 GPU.mode (GPU 运行模式)
    → 由 Scheduler 在任务启动时设置
```

**调度逻辑**:
```python
# 在 scheduler.py 中
if task.task_mode == TaskMode.EXCLUSIVE:
    # 需要 GPU 完全空闲
    if gpu.can_accept_task(TaskMode.EXCLUSIVE):
        # 设置 GPU 为独占模式
        gpu_manager.set_gpu_mode_for_task(gpu_id, task, TaskMode.EXCLUSIVE)
```

**使用场景**:
```python
# 创建共享任务
task = Task(
    script_path="test.py",
    task_mode=TaskMode.SHARED  # 可以与其他任务共享 GPU
)

# 创建独占任务
task = Task(
    script_path="benchmark.py",
    task_mode=TaskMode.EXCLUSIVE  # 需要独占 GPU
)

# 检查 GPU 是否可接受此模式的任务
if gpu.can_accept_task(task.task_mode):
    # 分配任务
```

---

### 4. TaskStatus - 任务状态

```python
class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

**用途**: 表示任务的当前执行状态

**状态转换**:
```
PENDING (刚提交)
   ↓
QUEUED (已分配GPU，等待执行)
   ↓
RUNNING (执行中)
   ↓
COMPLETED (成功完成)
FAILED (执行失败)
CANCELLED (被取消)
```

**详细说明**:

1. **PENDING (待处理)**
   - 任务刚被提交，在全局队列中
   - 还未分配 GPU
   - 可以被取消

2. **QUEUED (已排队)**
   - 已分配特定 GPU
   - 在 GPU 队列中等待执行
   - 可以被取消

3. **RUNNING (运行中)**
   - 任务正在执行
   - 有对应的进程 (subprocess.Popen)
   - 只能通过 force cancel 取消

4. **COMPLETED (已完成)**
   - 任务成功完成 (exit_code == 0)
   - 终态，不可取消
   - 保留日志和输出

5. **FAILED (失败)**
   - 任务执行失败 (exit_code != 0)
   - 或者超时、异常
   - 终态，不可取消
   - 保留错误信息和日志

6. **CANCELLED (已取消)**
   - 任务被用户或系统取消
   - 终态，不可恢复
   - 保留取消原因

**使用场景**:
```python
# 检查任务是否完成
if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
    print("Task finished")

# 检查任务是否可取消
if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
    task_queue.cancel_task(task_id)

# 筛选运行中的任务
running_tasks = [t for t in tasks if t.status == TaskStatus.RUNNING]
```

---

### 5. GPUStatus - GPU 状态

```python
class GPUStatus(str, Enum):
    """GPU status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"      # Severe error, all tasks paused
    MAINTENANCE = "maintenance"
```

**用途**: 表示 GPU 的可用性状态

**取值详解**:

1. **ONLINE (在线)**
   - GPU 正常工作，可以接受任务
   - 默认状态
   - `gpu.can_accept_task()` 返回 True

2. **OFFLINE (离线)**
   - GPU 暂时不可用
   - 不接受新任务
   - 可由管理员设置

3. **ERROR (错误)**
   - GPU 遇到严重错误
   - 触发 severe error 机制
   - 所有任务被终止并重新排队
   - 系统暂停调度 60 秒
   - 自动恢复或手动清除

4. **MAINTENANCE (维护)**
   - GPU 正在维护
   - 不接受新任务
   - 与 OFFLINE 类似，但语义不同

**调度影响**:
```python
# gpu_manager.py: find_available_gpu()
def find_available_gpu(...):
    if self.severe_error_active:
        return None  # 有 GPU 处于 ERROR 状态时，全局暂停
    
    for gpu in gpus:
        if gpu.status != GPUStatus.ONLINE:
            continue  # 跳过非 ONLINE 的 GPU
        # ...
```

**使用场景**:
```python
# 设置 GPU 为维护状态
gpu_manager.set_gpu_status(0, GPUStatus.MAINTENANCE)

# 检查 GPU 是否可用
if gpu.status == GPUStatus.ONLINE:
    # 可以分配任务

# 触发错误状态
gpu_manager.trigger_severe_error(0, "CUDA device error")
# GPU 0 变为 ERROR 状态，触发全局暂停
```

---

## 配置类

### ServerConfig - 服务器配置

```python
@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    log_file: str | None = "logs/nvgpu_server.log"
    log_level: str = "DEBUG"
    
    # Default GPU settings
    default_gpu_mode: GPUMode = GPUMode.SHARED
    default_memory_threshold: float = 0.75  # 75% memory usage threshold
    default_max_concurrent_tasks: int = 3   # Maximum concurrent tasks in shared mode
    
    # Task settings
    task_timeout: int = 600  # 10 minutes default timeout
    error_pause_duration: int = 60  # 1 minute pause on severe error (auto-resume)
    
    # Scheduler settings
    scheduler_interval: float = 1.0  # Check every 1 second
    gpu_monitor_interval: float = 5.0  # Monitor GPU every 5 seconds
```

**用途**: 集中管理服务器的所有配置参数

**字段详解**:

#### 1. 服务器配置

```python
host: str = "0.0.0.0"
```
- **含义**: FastAPI 服务器监听地址
- **默认值**: "0.0.0.0" (监听所有网络接口)
- **可选值**: "localhost", "127.0.0.1", 或特定 IP
- **影响**: 决定谁可以访问服务器

```python
port: int = 8080
```
- **含义**: FastAPI 服务器监听端口
- **默认值**: 8080
- **影响**: 客户端连接的端口号

```python
log_file: str | None = "logs/nvgpu_server.log"
```
- **含义**: 历史日志文件路径（追加模式）
- **默认值**: "logs/nvgpu_server.log"
- **特殊值**: None 表示不写历史日志
- **行为**: 所有会话日志追加到此文件

```python
log_level: str = "DEBUG"
```
- **含义**: 日志级别
- **可选值**: "DEBUG", "INFO", "WARNING", "ERROR"
- **默认值**: "DEBUG" (最详细)
- **影响**: 控制日志输出的详细程度

#### 2. GPU 默认设置

```python
default_gpu_mode: GPUMode = GPUMode.SHARED
```
- **含义**: 新注册 GPU 的默认模式
- **默认值**: SHARED
- **影响**: `gpu_manager.register_gpu()` 使用此值

```python
default_memory_threshold: float = 0.75
```
- **含义**: 默认内存使用阈值（75%）
- **范围**: 0.0 - 1.0
- **影响**: 超过阈值的 GPU 拒绝新 SHARED 任务
- **注意**: EXCLUSIVE 任务不受此限制

```python
default_max_concurrent_tasks: int = 3
```
- **含义**: SHARED 模式下默认最大并发任务数
- **默认值**: 3
- **影响**: `gpu.can_accept_task()` 检查此限制
- **注意**: EXCLUSIVE 模式此值无效（固定为 1）

#### 3. 任务设置

```python
task_timeout: int = 600
```
- **含义**: 任务执行超时时间（秒）
- **默认值**: 600 秒 (10 分钟)
- **行为**: 超时后任务被 kill，状态设为 FAILED
- **实现**: `task_runner.py: process.wait(timeout=config.task_timeout)`

```python
error_pause_duration: int = 60
```
- **含义**: 严重错误后的暂停时间（秒）
- **默认值**: 60 秒 (1 分钟)
- **行为**: 触发 severe error 后：
  1. 立即终止所有运行任务
  2. 暂停调度 60 秒
  3. 60 秒后自动恢复
- **实现**: `gpu_manager.py: _monitor_loop()`

#### 4. 调度器设置

```python
scheduler_interval: float = 1.0
```
- **含义**: 调度器检查间隔（秒）
- **默认值**: 1.0 秒
- **影响**: 任务被分配和启动的延迟
- **权衡**: 
  - 更小的值 → 更快响应，更高 CPU 使用
  - 更大的值 → 更慢响应，更低 CPU 使用

```python
gpu_monitor_interval: float = 5.0
```
- **含义**: GPU 监控更新间隔（秒）
- **默认值**: 5.0 秒
- **影响**: GPU 内存使用情况的更新频率
- **注意**: `find_available_gpu()` 会主动更新，不完全依赖此值

---

## 全局配置实例

```python
# Global server configuration instance
config = ServerConfig()
```

**用途**: 提供全局配置对象，所有模块共享

**使用方式**:
```python
from config import config

# 读取配置
timeout = config.task_timeout

# 修改配置（不推荐运行时修改）
config.task_timeout = 1200  # 改为 20 分钟
```

**注意事项**:
1. 配置在服务器启动时初始化
2. 运行时修改可能导致不一致
3. 建议通过环境变量或配置文件管理

---

## 设计考虑

### 1. 为什么使用 str + Enum?

```python
class GPUMode(str, Enum):
    EXCLUSIVE = "exclusive"
```

**优势**:
- 可以直接用作字符串：`mode.value` 或 `str(mode)`
- JSON 序列化友好
- 数据库存储方便
- 类型安全（IDE 支持）

**示例**:
```python
# 作为枚举使用
if gpu.mode == GPUMode.EXCLUSIVE:
    pass

# 作为字符串使用
json_data = {"mode": gpu.mode.value}  # "exclusive"

# HTTP 响应
return {"mode": str(gpu.mode)}  # "exclusive"
```

### 2. 为什么 TaskType 和 TaskMode 分离?

**TaskType (业务分类)**:
- 功能测试 vs 性能测试
- 用于统计、筛选、报表
- 可选字段

**TaskMode (技术需求)**:
- 独占 vs 共享
- 影响调度决策
- 必选字段（有默认值）

**好处**:
- 概念清晰，职责分离
- 灵活性：功能测试也可以要求独占
- 智能默认：根据 TaskType 推断 TaskMode

### 3. 配置的可扩展性

使用 `@dataclass` 便于：
- 添加新字段
- IDE 自动补全
- 类型检查

未来可以支持：
- 从配置文件加载
- 环境变量覆盖
- 动态更新

---

## 常见使用模式

### 1. 创建任务时应用智能默认值

```python
# 在 api_server.py 中
def create_task(task_submit: TaskSubmit) -> Task:
    # 智能默认值在 Task.__post_init__() 中应用
    task = Task(
        script_path=task_submit.script_path,
        task_type=TaskType(task_submit.task_type) if task_submit.task_type else None,
        # task_mode 由 __post_init__() 根据 task_type 设置
    )
    return task
```

### 2. 检查 GPU 是否可接受任务

```python
# 在 gpu_manager.py 中
def can_gpu_accept_task(gpu: GPU, task_mode: TaskMode) -> bool:
    # 首先检查状态
    if gpu.status != GPUStatus.ONLINE:
        return False
    
    # 然后检查模式兼容性
    return gpu.can_accept_task(task_mode)
```

### 3. 状态转换

```python
# 在 task_runner.py 中
def run_task(task: Task, gpu_id: int):
    task.status = TaskStatus.RUNNING  # QUEUED → RUNNING
    try:
        exit_code = process.wait()
        if exit_code == 0:
            task.status = TaskStatus.COMPLETED  # RUNNING → COMPLETED
        else:
            task.status = TaskStatus.FAILED  # RUNNING → FAILED
    except subprocess.TimeoutExpired:
        task.status = TaskStatus.FAILED  # RUNNING → FAILED (超时)
```

---

## 相关文档

- [数据模型详解](CODE_REFERENCE_MODELS.md) - 如何使用这些枚举
- [调度器详解](CODE_REFERENCE_SCHEDULER.md) - 状态转换的实现
- [GPU 管理器详解](CODE_REFERENCE_GPU_MANAGER.md) - GPU 状态管理

---

**下一步**: 阅读 [数据模型详解](CODE_REFERENCE_MODELS.md) 了解如何使用这些枚举定义数据结构。
