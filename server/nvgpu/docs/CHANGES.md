# NVGPU Server 变更说明

**主要版本变更和功能演进**

---

## 当前版本特性

### 核心设计：三参数分离

```python
client.submit_task(
    "test.py",
    task_mode="shared",      # GPU行为控制（智能默认）
    task_type="functional",  # 业务分类（可选）
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"  # 具体标识（可选）
)
```

### 智能默认值

```
task_type="functional"   → task_mode="shared"     ✓
task_type="performance"  → task_mode="exclusive"  ✓
task_type="both"         → task_mode="exclusive"  ✓
task_type=None           → task_mode="shared"     ✓
```

---

## 主要变更历史

### 概念分离设计

**变更内容:**
1. **task_mode** - GPU行为控制
   - `exclusive` 或 `shared`
   - 智能默认（基于`task_type`）

2. **task_type** - 业务分类
   - `functional`, `performance`, `both`
   - 用于统计和筛选

3. **task_label** - 具体标识
   - 任意字符串
   - 用于精确识别

**设计理念:**
- 概念清晰分离
- 智能默认值减少输入
- 灵活性与易用性兼顾

---

### 任务驱动GPU模式

**功能:**
- GPU模式根据任务需求自动切换
- 任务开始时设置模式，结束时恢复
- 保留手动模式覆盖能力

**优点:**
- 自动化管理
- 减少手动干预
- 提高资源利用率

---

### 严重错误处理增强

**机制:**
1. 检测到GPU严重错误时
2. 强制终止所有运行中的任务（SIGTERM → SIGKILL）
3. 被终止的任务重新插入队列最前面
4. 系统暂停60秒后自动恢复

**优点:**
- 自动恢复
- 任务不丢失
- 保证系统稳定性

---

### 日志系统增强

**双日志系统:**
- 会话日志：`nvgpu_server_YYYYMMDD_HHMMSS.log`
- 历史日志：`nvgpu_server.log`（追加）

**任务日志:**
- 独立的stdout/stderr文件
- 支持大文件（无大小限制）
- 优雅处理segment fault和core dump

**时间统计:**
- 总耗时：从提交到完成
- 执行时间：从开始到结束
- 毫秒级精度

---

### GPU资源管理优化

**并发控制:**
- `max_concurrent_tasks`: 限制并发任务数
- `memory_threshold`: 内存使用阈值
- 双重保护机制

**实时监控:**
- 任务分配前主动更新GPU内存
- 确保准确的调度决策
- 处理延迟内存分配的任务

---

### 任务取消功能

**实现:**
- 支持取消等待中和运行中的任务
- 优雅终止（SIGTERM）+ 强制终止（SIGKILL）
- 完整的状态跟踪

**使用:**
```python
client.cancel_task(task_id)
```

---

### work_dir功能

**功能:**
- 指定任务的工作目录
- 支持相对路径和绝对路径
- 便捷方法：`submit_task_in_script_dir`

**使用:**
```python
# 方式1: 显式指定
client.submit_task("test.py", work_dir="/path/to/dir")

# 方式2: 使用脚本目录
client.submit_task_in_script_dir("/path/to/script.py")
```

---

### 绝对路径支持

**改进:**
- 所有路径相对于main.py的父目录
- 支持从任意目录启动服务器
- 配置文件路径智能解析

---

### GPU配置文件

**功能:**
- YAML格式配置
- GPU硬件信息
- ID映射（nvidia-smi ↔ CUDA_VISIBLE_DEVICES）
- 自动注册

**文件位置:**
```
configs/gpu_resource.yml
```

---

## API变更

### 任务提交

**当前API:**
```python
client.submit_task(
    script_path: str,
    task_mode: str | None = None,
    task_type: str | None = None,
    task_label: str | None = None,
    work_dir: str = ".",
    args: List[str] | None = None,
    env: Dict[str, str] | None = None,
    gpu_id: int | None = None
) -> str
```

**简化用法:**
```python
# 最简单
client.submit_task("test.py")

# 带分类
client.submit_task("test.py", task_type="functional")

# 完整
client.submit_task(
    "test.py",
    task_type="functional",
    task_label="xpiler_cuda/test"
)
```

---

### GPU管理

**新增功能:**
- 手动模式设置/清除
- 并发任务数配置
- 内存阈值配置

**API:**
```python
# 设置手动模式
client.set_gpu_mode(0, "exclusive", manual=True)

# 清除手动模式
client.clear_gpu_manual_mode(0)

# 配置参数
client.configure_gpu(0, memory_threshold=0.8, max_concurrent_tasks=6)
```

---

## 性能改进

### 调度效率
- 实时GPU内存更新
- 智能任务分配
- 减少调度延迟

### 资源利用
- Shared模式并发控制
- 内存阈值保护
- 防止GPU过载

### 日志性能
- 异步日志写入
- 按需读取（offset/limit支持）
- 大文件优化

---

## 向后兼容性

### 完全兼容
所有旧版API仍然有效：
```python
# 旧版写法仍然工作
client.submit_task("test.py", task_type="functional", task_mode="shared")

# 但推荐新写法
client.submit_task("test.py", task_type="functional")
```

### 弃用通知
无弃用项，所有功能都得到支持。

---

## 未来计划

参考 [TODO.md](TODO.md) 获取详细的开发计划。

---

## 相关文档

- [设计文档](DESIGN.md) - 完整设计说明
- [快速入门](QUICKSTART.md) - 快速上手
- [API参考](API_REFERENCE.md) - 详细API文档

---

**最后更新:** 2025-01-15

