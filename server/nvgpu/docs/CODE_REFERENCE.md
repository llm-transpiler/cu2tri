# NVGPU Server 核心模块代码详解

本文档系列详细解释 NVGPU Server 核心模块的代码实现，包括每个函数的作用、代码逻辑、前提条件和副作用。

**版本:** v2.5  
**最后更新:** 2025-01-15

---

## 文档结构

本代码参考手册分为以下几个部分：

1. **[配置模块 (config.py)](CODE_REFERENCE_CONFIG.md)** - 枚举定义和服务器配置
2. **[数据模型 (models.py)](CODE_REFERENCE_MODELS.md)** - GPU 和 Task 数据结构
3. **[GPU 管理器 (gpu_manager.py)](CODE_REFERENCE_GPU_MANAGER.md)** - GPU 资源管理和监控
4. **[任务队列 (task_queue.py)](CODE_REFERENCE_TASK_QUEUE.md)** - 任务队列管理
5. **[任务执行器 (task_runner.py)](CODE_REFERENCE_TASK_RUNNER.md)** - 任务进程执行
6. **[调度器 (scheduler.py)](CODE_REFERENCE_SCHEDULER.md)** - 任务调度逻辑
7. **[Python 客户端 (client.py)](CODE_REFERENCE_CLIENT.md)** - API 客户端库

---

## 模块依赖关系

```
config.py (基础配置)
    ↓
models.py (数据模型)
    ↓
gpu_manager.py ←→ task_queue.py ←→ task_runner.py
    ↓               ↓                   ↓
    └───────→ scheduler.py ←────────────┘
                  ↓
            api_server.py (REST API)
                  ↓
              client.py (客户端)
```

### 依赖说明

- **config.py**: 独立模块，定义所有枚举和配置
- **models.py**: 依赖 config.py，定义数据结构
- **gpu_manager.py**: 管理 GPU 资源，依赖 task_queue 和 task_runner 进行错误处理
- **task_queue.py**: 管理任务队列，依赖 task_runner 进行强制取消
- **task_runner.py**: 执行任务进程，相对独立
- **scheduler.py**: 协调以上三个模块，实现任务调度
- **client.py**: 独立的客户端库，通过 HTTP 与服务器通信

---

## 核心概念速查

### 任务状态转换

```
PENDING (提交) → QUEUED (分配GPU) → RUNNING (执行中)
                                       ↓
                                    COMPLETED (成功)
                                    FAILED (失败)
                                    CANCELLED (取消)
```

### GPU 模式

- **EXCLUSIVE (独占)**: 同时只能运行一个任务
- **SHARED (共享)**: 可运行多个任务（受内存和并发数限制）

### 任务模式

- **EXCLUSIVE (独占任务)**: 需要独占 GPU
- **SHARED (共享任务)**: 可与其他任务共享 GPU

### GPU 模式管理

1. **手动模式 (manual_mode)**: 管理员设置，最高优先级
2. **任务锁定 (mode_locked_by)**: 独占任务运行时锁定
3. **默认模式**: shared

---

## 线程安全

### 使用的锁

所有核心模块使用 `threading.RLock()` (可重入锁) 保证线程安全：

- **GPUManager.lock**: 保护 GPU 状态
- **TaskQueue.lock**: 保护任务队列
- **TaskRunner.lock**: 保护运行进程字典

### 锁的使用原则

1. **锁的粒度**: 细粒度锁，只在必要时持有
2. **避免死锁**: 注意锁的获取顺序
3. **外部操作**: 某些操作（如进程管理）在锁外执行

---

## 关键设计决策

### 1. 调度器 Race Condition 修复

**问题**: 在多线程环境下，GPU 状态检查和任务启动之间有时间窗口

**解决**: 在主调度线程中标记任务为 RUNNING，然后再启动执行线程

```python
# scheduler.py: _schedule_round()
# 关键顺序：
1. 检查 GPU 是否可接受任务 (在锁内)
2. 标记任务为 RUNNING (在锁内，主线程)
3. 设置 GPU 模式 (在锁内，主线程)
4. 启动执行线程 (锁外)
```

### 2. Severe Error 处理

**设计**: 当 GPU 出现严重错误时：
1. 标记 GPU 为 ERROR 状态
2. 强制终止所有运行中的任务
3. 将被终止的任务重新排队（队首）
4. 暂停调度 60 秒后自动恢复

**实现位置**: `gpu_manager.py: trigger_severe_error()`

### 3. GPU 内存实时更新

**设计**: 在调度决策前更新 GPU 内存使用情况

**原因**: 某些任务启动后延迟分配 GPU 内存，调度器需要最新数据

**实现位置**: `gpu_manager.py: find_available_gpu()`

---

## 日志系统

### 日志级别使用

- **ERROR**: 严重错误，需要人工介入
- **WARNING**: 警告，可能影响功能但不致命
- **INFO**: 重要操作和状态变更
- **DEBUG**: 详细调试信息

### 日志模块

每个核心模块有独立的 logger：

```python
logger = setup_logger("module_name")
```

模块名称：
- `gpu_manager` - GPU 管理
- `task_queue` - 任务队列
- `task_runner` - 任务执行
- `scheduler` - 调度器

---

## 代码约定

### 类型注解

使用 Python 3.12 风格类型注解：

```python
# ✓ 正确
def get_gpu(self, gpu_id: int) -> GPU | None:
    pass

# ✗ 旧风格 (已弃用)
def get_gpu(self, gpu_id: int) -> Optional[GPU]:
    pass
```

### 命名约定

- **类名**: PascalCase (`GPUManager`, `TaskQueue`)
- **函数名**: snake_case (`submit_task`, `find_available_gpu`)
- **常量**: UPPER_CASE (`NVML_AVAILABLE`, `NVGPU_ROOT`)
- **私有方法**: 前缀 `_` (`_update_gpu_memory`, `_schedule_round`)

### 文档字符串

所有公共 API 必须有文档字符串：

```python
def submit_task(self, task: Task) -> str:
    """Submit a new task.
    
    Args:
        task: Task object to submit
        
    Returns:
        Task ID
        
    Side Effects:
        - Adds task to global queue
        - Sets task status to PENDING
        - Logs submission
    """
```

---

## 性能考虑

### 1. 锁竞争

- 锁持有时间尽可能短
- 耗时操作（如文件 I/O、进程管理）在锁外执行

### 2. 内存管理

- 大文件使用流式读取
- 日志预览限制大小（10KB）
- stdout/stderr 分别存储

### 3. 并发控制

- 调度器每秒运行一次（可配置）
- GPU 监控每 5 秒更新（可配置）
- 任务超时 600 秒（可配置）

---

## 错误处理策略

### 1. 任务级错误

- 任务失败不影响其他任务
- 退出码和错误信息记录到任务对象
- stderr 内容保存到文件

### 2. GPU 级错误

- 单个 GPU 错误不影响其他 GPU
- 错误状态可手动或自动恢复

### 3. 系统级错误

- Severe error 触发全局暂停
- 自动重试机制（任务重新排队）
- 监控线程独立运行，避免单点故障

---

## 测试覆盖

- **单元测试**: 108 个测试用例，100% 通过
- **代码覆盖率**: 70% 总体覆盖率
- **关键模块覆盖**: 
  - models.py: 92%
  - task_queue.py: 88%
  - scheduler.py: 87%

详见: [测试覆盖报告](../tests/TEST_COVERAGE.md)

---

## 下一步

选择一个模块开始深入学习：

- 从 **[配置模块](CODE_REFERENCE_CONFIG.md)** 开始了解基础概念
- 从 **[数据模型](CODE_REFERENCE_MODELS.md)** 了解数据结构
- 从 **[调度器](CODE_REFERENCE_SCHEDULER.md)** 了解核心调度逻辑
- 从 **[客户端](CODE_REFERENCE_CLIENT.md)** 了解如何使用 API

---

## 相关文档

- [API 参考手册](API_REFERENCE_ZH.md) - REST API 完整文档
- [设计文档](DESIGN.md) - 系统设计理念
- [快速入门](QUICKSTART.md) - 使用指南

---

**维护者注意**: 修改核心模块代码后，请同步更新对应的代码参考文档。

