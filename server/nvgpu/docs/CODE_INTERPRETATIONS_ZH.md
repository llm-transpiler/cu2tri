# 代码解读记录（持续更新）

- 用途：集中记录对仓库内文件的代码解读/笔记，后续可持续追加。
- 约定：每条解读以“日期 + 文件路径”为标题；可包含“焦点片段”、“结构与职责”、“关键方法速览（带起始行）”、“常见用法流程”、“与其他组件交互”、“可进一步展开”。
- 追加规则：新的解读按时间顺序向下追加，或在顶部追加最新一条。

---

## 2025-10-09 server/nvgpu/gpu_manager.py

- 焦点片段：`_initialize_nvml`
- 目的：集中管理 GPU 资源，负责注册/状态/模式管理、NVML 监控、任务占用跟踪、严重错误处理与后台监控线程。
- 依赖：`pynvml`（可选）、`models.GPU/GPUStatus/GPUMode`、`config.config`、`logger`。
- 并发：使用 `threading.RLock()` 保护对共享状态（如 `self.gpus`）的访问。

### 文件概览
- 职责：面向任务调度的 GPU 资源抽象与运行时状态维护；在可用时通过 NVML 采集显存占用，参与调度决策。
- NVML 不可用时：仅降级监控（显存占用不更新），系统仍可运行，但调度无法基于实时显存负载。

### 初始化与 NVML
- 构造函数：完成字段初始化并调用 NVML 初始化。`server/nvgpu/gpu_manager.py:25`
- NVML 初始化：检测 `pynvml` 可用性，`pynvml.nvmlInit()` 成功后设置 `self.nvml_initialized` 并记录 GPU 数量；失败仅降级。`server/nvgpu/gpu_manager.py:53`

### GPU 注册与状态
- 注册 GPU：按配置默认值创建 `GPU`（模式、显存阈值、共享并发上限），加入管理表。`server/nvgpu/gpu_manager.py:68`
- 取消注册：若仍有运行任务则拒绝。`server/nvgpu/gpu_manager.py:88`
- 设置状态：支持 ONLINE/OFFLINE/MAINTENANCE/ERROR。`server/nvgpu/gpu_manager.py:104`
- 查询：`get_gpu`、`list_gpus` 提供只读查询。`server/nvgpu/gpu_manager.py:262`, `server/nvgpu/gpu_manager.py:267`

### 模式管理（手动/任务驱动）
- 设定模式：`set_gpu_mode` 支持 `manual=True` 设为手动模式（优先级最高），否则仅改当前模式。`server/nvgpu/gpu_manager.py:114`
- 清除手动模式：恢复为受任务驱动或默认共享。`server/nvgpu/gpu_manager.py:140`
- 任务开始设定：EXCLUSIVE 任务锁定 `mode_locked_by=task_id`；SHARED 在未被他人锁定时切共享；存在手动模式时始终尊重手动模式。`server/nvgpu/gpu_manager.py:157`
- 任务结束恢复：若持锁任务结束则解锁；若有手动模式则恢复之，否则在无运行任务时恢复共享。`server/nvgpu/gpu_manager.py:197`

### 调度与显存更新
- 查找可用 GPU：严重错误期间直接拒绝；优先尝试指定 GPU；否则先刷新全部显存后按 `GPU.can_accept_task` 选择。`server/nvgpu/gpu_manager.py:272`
- 更新显存占用：NVML 可用时 `nvmlDeviceGetMemoryInfo`；支持通过全局 `gpu_config_loader` 映射逻辑 id 到 `nvidia-smi` index。`server/nvgpu/gpu_manager.py:400`
- 动态约束：`set_gpu_memory_threshold`、`set_gpu_max_concurrent_tasks` 可动态调整阈值与并发上限。`server/nvgpu/gpu_manager.py:234`, `server/nvgpu/gpu_manager.py:248`
- 降级行为：NVML 不可用则 `current_memory_usage` 不更新（默认 0.0），调度主要受并发上限影响。

### 任务运行跟踪
- 运行/完成标记：维护 `GPU.running_tasks` 列表，用于并发与模式兼容性判断。`server/nvgpu/gpu_manager.py:316`, `server/nvgpu/gpu_manager.py:326`

### 严重错误处理（全局暂停与自动恢复）
- 触发严重错误：将 GPU 置 ERROR，记录错误时间；终止该 GPU 上所有运行任务并按原顺序重新入队队首（重试）；清空 `running_tasks`。`server/nvgpu/gpu_manager.py:338`
- 自动恢复：后台监控在 `config.error_pause_duration` 秒后自动清除严重错误状态；也可手动 `clear_severe_error`。`server/nvgpu/gpu_manager.py:393`, `server/nvgpu/gpu_manager.py:423`

### 后台监控线程
- 监控循环：周期刷新全部 GPU 显存；超时后自动恢复严重错误状态；睡眠间隔由 `config.gpu_monitor_interval` 控制。`server/nvgpu/gpu_manager.py:423`
- 启停：`start_monitoring` 启动守护线程，`stop_monitoring` 请求停止并等待退出。`server/nvgpu/gpu_manager.py:445`, `server/nvgpu/gpu_manager.py:456`
- 关闭：`shutdown` 停监控并在已初始化 NVML 时调用 `pynvml.nvmlShutdown()`。`server/nvgpu/gpu_manager.py:463`

### 与其他组件的交互
- `models.GPU.can_accept_task`：调度核心规则（状态、模式兼容、并发、显存阈值）；由 `find_available_gpu` 调用。
- `task_queue` / `task_runner`：严重错误处理时用于终止与重排队任务（通过 `set_dependencies` 注入）。`server/nvgpu/gpu_manager.py:42`
- `gpu_config_loader`（全局）：将逻辑 GPU id 映射为 `nvidia-smi` index，便于与部署环境一致。`server/nvgpu/gpu_manager.py:400`

### 常见用法流程
1) 注册 GPU → 2) 启动监控 → 3) 调度器调用 `find_available_gpu` → 4) 任务开始：`set_gpu_mode_for_task` + `mark_task_running` → 5) 任务结束：`mark_task_completed` + `restore_gpu_mode_after_task` → 6) 如遇严重错误：`trigger_severe_error` 后自动/手动恢复。

### 可进一步展开
- 更细的 `GPU.can_accept_task` 判定逻辑与边界情形。
- `gpu_config_loader` 的映射配置样例与在多机/异构部署下的使用策略。
- 端到端任务流（入队→调度→运行→回收）的调用路径与日志字段对照。

---

（后续解读可按相同格式在本文档中追加新条目。）

