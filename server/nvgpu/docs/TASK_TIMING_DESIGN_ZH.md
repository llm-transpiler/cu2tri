# 任务生命周期计时设计

本文档详细说明 `server/nvgpu` 中任务生命周期计时（Task Timing）的设计与实现。目标是让调度链路的关键阶段具备**统一、可量化的毫秒耗时数据**，同时保留原有的 datetime 日志时间戳，避免对现有日志分析流程造成破坏。

## 1. 背景与目标

在改造之前，控制面仅通过以下方式衡量耗时：

- `Task.submit_time / queued_time / start_time / end_time` 等字段记录 datetime，用于在日志或 API 中展示事件时间点；
- 各阶段耗时通过 datetime 相减得出，缺少更细粒度的采样能力；
- `task_runner.py` 使用 `HostTimer` 仅监控 `process.wait`，未覆盖排队、启动等前置阶段。

本次设计的目标：

1. **引入通用计时器**：覆盖“提交 → GPU 分配 → 队列 → 执行 → 完成”的完整生命周期；
2. **跨模块协作**：计时操作分布在 `TaskQueue`、`Scheduler`、`TaskRunner` 等组件，需要一个可以在不同作用域开启/关闭的工具；
3. **兼容性**：保留 datetime 时间点与基于时间差的兼容逻辑，以免影响已有接口和日志解析；
4. **日志增强**：在任务摘要日志中展示新的毫秒级统计项，方便排查性能瓶颈。

## 2. 计时工具：`TaskTimer`

核心实现位于 `server/nvgpu/task_timer.py`，封装了一个轻量级的 `HostTimer` 管理器：

| 方法 | 描述 |
| ---- | ---- |
| `start(label)` | 在给定标签下开启计时段，无视重复开启请求 |
| `stop(label)` | 终止计时段并返回累计耗时（毫秒），若未开启则返回 `None` |
| `get_duration_ms(label)` / `as_dict()` | 按标签读取或导出全部耗时数据 |
| `reset()` | 停止所有活动上下文并清空采样数据 |

内部通过 `_ManagedContext` 手动管理上下文的进入/退出，确保计时与业务流程解耦，跨线程/跨函数使用时无需显式 `with` 块。

### 计时标签

- `total`：提交至结束的总体耗时；
- `waiting`：提交后等待执行的总耗时（队列 + GPU 分配）；
- `pending`：处于全局待处理队列的时间；
- `queue`：被分配到指定 GPU 队列后的排队时间；
- `running`：子进程生命周期（执行开始 → 结束/终止）的时间；
- 历史上曾使用 `execution` 标签，代码仍保留兼容逻辑以应对迁移期产生的数据；
- 其余标签可在未来扩展，例如 Scheduler 内部阶段或 GPU 管理事件。

## 3. 生命周期采集流程

### 3.1 任务提交 (`TaskQueue.submit_task`)

1. 将任务放入全局队列时，启动 `total`、`waiting`、`pending` 三个计时段；
2. 该阶段后续在 `queue_task_for_gpu` 停止 `pending`，平均衡量“提交到 GPU 分配”时间。

### 3.2 任务入 GPU 队列 (`TaskQueue.queue_task_for_gpu`)

1. 记录 `queued_time` datetime；
2. 停止 `pending` 计时并写入 `phase_timing_ms["pending"]`；
3. 随即开启 `queue` 计时，统计在 GPU 专属队列中的排队耗时。

### 3.3 任务开始执行 (`TaskRunner.run_task`)

1. 切换状态为 `RUNNING`、设置 `start_time`；
2. 停止 `queue` 与 `waiting` 计时，并更新对应 `phase_timing_ms`；
3. 启动 `running` 计时，在子进程生命周期内累计耗时；
4. `process.wait` 仍然使用原有 `HostTimer` 采样写入 `task.host_timing_ms`，与新设计互补。

### 3.4 任务收尾 (`TaskRunner.run_task` 的 finally)

1. 停止 `running` 计时并写入 `phase_timing_ms["running"]`；
2. 停止 `total` 计时，得出完整生命周期耗时。

同一代码块同时保持 datetime 记录，用于日志与 API 的时间点输出。

### 3.5 取消或重排任务

- `TaskQueue.cancel_task`：调用 `_finalize_timing_on_cancel` 停止当前状态涉及的计时段（如 `pending` / `queue`），并关闭 `waiting`、`total`；
- `TaskQueue.push_front`：任务重新进入全局队列时，重新开启 `pending`，使票据累计时间中断而非叠加；
- 强制取消（`force_cancel_task`）依赖 `TaskRunner` 在 finally 中收尾 `running` 和 `total`，确保异常路径也有完整统计。

## 4. 数据结构与输出

### 4.1 `Task` 数据类 (`server/nvgpu/models.py`)

- 新增字段：
  - `phase_timing_ms: dict[str, float]`：存储各标签的毫秒耗时；
  - `timer: TaskTimer`：实际计时器实例（`repr=False` 避免日志噪音）。
- 各阶段属性（`pending_time_ms` 等）优先使用 `phase_timing_ms`，若计时器缺失则回退原有 datetime 差值逻辑，保持兼容性。
- `to_dict()` 输出中保留 datetime 及衍生毫秒数，API 调用方无需修改。

### 4.2 任务日志 (`server/nvgpu/task_runner.py::_write_log_file`)

日志仍按原格式输出 datetime，与新增计时项并行：

```
=== Timing ===
Submit Time:  2024-05-20 10:00:00+08:00
Queued Time:  2024-05-20 10:00:01+08:00
Start Time:   2024-05-20 10:00:03+08:00
End Time:     2024-05-20 10:00:05+08:00

=== Timing Breakdown (milliseconds) ===
Pending Time:          950.32 ms
Queue Time:           2010.57 ms
Total Waiting:        2960.89 ms
Running Time:        2025.61 ms
Total Time:           4986.50 ms
```

`host_timing_ms` 区块继续展示 `process.wait` 等宿主侧采样，帮助区分流程耗时与子进程内部耗时。

## 5. 组件职责总结

| 模块 | 变更点 | 作用 |
| ---- | ---- | ---- |
| `task_timer.py` | 新增 `TaskTimer` | 提供跨模块通用计时器 |
| `models.py` | 扩展 `Task` 字段及计算优先级 | 存储与导出计时结果 |
| `task_queue.py` | 在提交/入队/取消/重排时启动或结束计时 | 覆盖等待阶段 |
| `task_runner.py` | 运行时启停 `running`/`total`，保持日志输出 | 覆盖执行阶段与总体耗时 |

## 6. 扩展与注意事项

- **扩展标签**：未来可在 `Scheduler` 中增加“调度循环耗时”、或在 GPU 管理逻辑中监控 `set_gpu_mode_for_task` 等操作，只需调用 `task.timer.start/stop`；
- **并发安全**：`TaskTimer` 基于 `HostTimer` 实现，线程安全取决于调用顺序。当前调用均在主调度线程或任务执行线程中顺序发生，未出现并发读写；
- **序列化兼容**：`TaskTimer` 不会泄露到 API/日志序列化输出中。若通过持久化层存储 `Task`，需要忽略该字段；
- **异常路径**：`finally` 块与取消逻辑保证计时器无论成功、失败、超时都能落盘；若新增异常分支需确保调用 stop。

## 7. 验证建议

1. **功能测试**：运行 `server/nvgpu/run_tests.sh` 或 pytest，确认新字段未破坏现有测试；
2. **手动验证**：提交任务后查看 `logs/tasks/<task_id>.log`，确认毫秒级耗时与 datetime 对应；
3. **API 检查**：调用 `GET /tasks/{task_id}`，检查返回数据中 `pending_time_ms` 等字段是否填充。

通过以上设计，NVGPU Server 获得统一的生命周期耗时采集能力，为调度性能分析和优化提供更细粒度的依据。
