[X] 任意时刻发生严重错误的时候都应该将所有正在运行的任务直接强制kill然后重新加回任务队列最前方
[X] exclusive和shared模式的切换应该发生在task实际开始运行的时候，不应该手动设置，也就是说将现在的task的属性改成shared/exclusive，然后task启动的时候自动设置GPU Mode, 保留现在手动更改的接口

## 实现说明 (Implementation Notes)

### 1. 严重错误处理 (Severe Error Handling)
**实现位置:** `gpu_manager.py::trigger_severe_error()`

**机制:**
- 当触发严重错误时，系统会：
  1. 获取所有正在运行的任务ID
  2. 调用 `task_runner.kill_task()` 强制终止每个任务（SIGTERM → SIGKILL）
  3. 重置任务状态（start_time、end_time、exit_code等）
  4. 使用 `task_queue.push_front()` 将任务重新插入队列最前方（逆序以保持原顺序）
  5. 清空GPU的running_tasks列表
  6. 系统暂停60秒后自动恢复

**关键代码:**
```python
# gpu_manager.py Line 222-275
def trigger_severe_error(self, gpu_id: int, error_msg: str):
    # Kill all running tasks
    for task_id in running_task_ids:
        if self.task_runner.kill_task(task_id):
            killed_tasks.append(task)
    
    # Requeue at front (reverse order to maintain original order)
    for task in reversed(killed_tasks):
        task.start_time = None
        task.end_time = None
        task.exit_code = None
        self.task_queue.push_front(task)
```

### 2. 任务驱动的GPU模式切换 (Task-Driven GPU Mode Switching)
**实现位置:** `models.py`, `scheduler.py`, `gpu_manager.py`

**架构变更:**
- 添加了 `TaskMode` 枚举 (exclusive/shared) - 控制GPU行为
- 保留了 `TaskType` 枚举 (functional/performance) - 仅用于分类元数据
- GPU 模型添加了 `manual_mode` 和 `mode_locked_by` 字段

**模式切换机制:**
1. **任务提交时:** 用户指定 `task_mode` (exclusive/shared)
2. **任务开始时:** 调度器调用 `set_gpu_mode_for_task()`
   - 如果任务是exclusive，GPU切换到EXCLUSIVE模式并锁定
   - 如果任务是shared且GPU未被锁定，GPU切换到SHARED模式
   - 如果GPU有manual_mode，则尊重manual_mode不改变
3. **任务结束时:** 调度器调用 `restore_gpu_mode_after_task()`
   - 如果任务锁定了GPU，解锁
   - 恢复到manual_mode（如果有）或默认SHARED模式

**关键代码:**
```python
# scheduler.py Line 66-99
def _execute_task(self, task, gpu_id: int):
    self.gpu_manager.mark_task_running(gpu_id, task)
    # Set GPU mode based on task requirements
    self.gpu_manager.set_gpu_mode_for_task(gpu_id, task, task.task_mode)
    try:
        success = self.task_runner.run_task(task, gpu_id)
    finally:
        # Restore GPU mode after task completes
        self.gpu_manager.restore_gpu_mode_after_task(gpu_id, task.task_id)
        self.gpu_manager.mark_task_completed(gpu_id, task)
```

**手动模式接口保留:**
- `PUT /gpus/{id}/mode` - 设置manual mode (manual=true)
- `DELETE /gpus/{id}/mode` - 清除manual mode，允许任务驱动切换

### 文档更新
- [X] API Reference 完整文档: `docs/API_REFERENCE.md`
- [ ] 设计文档需要更新以反映新架构
- [ ] 示例脚本需要更新以使用新的task_mode参数
