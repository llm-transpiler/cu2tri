# NVGPU Server 设计文档

**GPU 任务调度服务器完整设计说明**

---

## 🎯 核心概念

### 三个参数，清晰分离

```python
client.submit_task(
    "test.py",
    task_mode="shared",      # GPU行为控制 (智能默认)
    task_type="functional",  # 业务分类 (可选)
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"  # 具体标识 (可选)
)
```

#### 1. **task_mode** - GPU行为控制
- **用途:** 控制GPU如何执行任务
- **取值:** `"exclusive"` 或 `"shared"`
- **默认:** 智能默认（基于`task_type`）
- **何时指定:** 需要覆盖智能默认值时

#### 2. **task_type** - 业务分类
- **用途:** 业务层面的分类，用于统计和筛选
- **取值:** 
  - `"functional"` - 功能测试
  - `"performance"` - 性能测试
  - `"both"` - 两者兼有
- **默认:** `None`（可选）
- **何时指定:** 需要分类统计时

#### 3. **task_label** - 具体标识
- **用途:** 具体的测试标签，精确识别
- **取值:** 任意字符串（建议使用层次结构）
- **示例:**
  - `"xpiler_cuda/add_3_3_256/cuda_vs_triton"`
  - `"matmul/4096x4096/baseline_vs_optimized"`
  - `"nightly_regression/test_suite_1/case_42"`
- **默认:** `None`（可选）
- **何时指定:** 需要精确识别和追踪时

---

## 💡 智能默认值

### 规则

```python
if task_mode is None:
    if task_type == "functional":
        task_mode = "shared"
    elif task_type == "performance":
        task_mode = "exclusive"
    elif task_type == "both":
        task_mode = "exclusive"  # 包含性能测试，需要独占
    else:
        task_mode = "shared"  # 安全默认
```

### 使用场景

```python
# 场景1: 只关心执行（90%的情况）
client.submit_task("test.py")  # → shared mode

# 场景2: 需要分类统计
client.submit_task("test.py", task_type="functional")  # → shared mode

# 场景3: 需要精确识别
client.submit_task(
    "test.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)  # → shared mode

# 场景4: 性能测试
client.submit_task("bench.py", task_type="performance")  # → exclusive mode

# 场景5: 覆盖默认值（特殊需求）
client.submit_task(
    "memory_test.py",
    task_mode="exclusive",      # 显式指定
    task_type="functional",
    task_label="stress_test/memory_limit"
)
```

---

## 📊 使用示例

### 1. 简单功能测试

```python
# 最简单
client.submit_task("test.py")

# 稍微详细
client.submit_task("test.py", task_type="functional")
```

### 2. 带标签的功能测试

```python
client.submit_task(
    "test_add.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)
```

### 3. 性能基准测试

```python
client.submit_task(
    "benchmark.py",
    task_type="performance",
    task_label="matmul/4096x4096/baseline"
)
# 自动使用exclusive模式
```

### 4. 综合测试

```python
client.submit_task(
    "comprehensive_test.py",
    task_type="both",
    task_label="nightly_regression/full_suite/v2.5"
)
# 自动使用exclusive模式（包含性能测试）
```

### 5. 特殊需求（覆盖默认）

```python
# 功能测试但需要独占GPU（测试显存极限）
client.submit_task(
    "memory_stress.py",
    task_mode="exclusive",  # 显式覆盖
    task_type="functional",
    task_label="stress_test/memory_limit"
)
```

---

## 🔍 过滤和查询

### 按task_type过滤

```python
# 查看所有功能测试
functional_tasks = client.list_tasks(task_type="functional")

# 查看所有性能测试
perf_tasks = client.list_tasks(task_type="performance")
```

### 按task_label过滤

```python
# 查看特定测试
cuda_tests = client.list_tasks(task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton")

# 查看某个测试套件
nightly_tests = client.list_tasks(task_label="nightly_regression/*")
```

### 组合过滤

```python
# 已完成的性能测试
completed_perf = client.list_tasks(
    status="completed",
    task_type="performance"
)
```

---

## 📝 标签命名建议

### 层次结构

推荐使用 `/` 分隔的层次结构：

```
<项目>/<测试名称>/<子分类>
```

### 示例

```python
# XPiler CUDA测试
"xpiler_cuda/add_3_3_256/cuda_vs_triton"
"xpiler_cuda/matmul_1024/baseline_vs_optimized"

# 回归测试
"nightly_regression/test_suite_1/case_001"
"nightly_regression/test_suite_1/case_002"

# 性能基准
"benchmark/matmul/4096x4096/fp32"
"benchmark/matmul/4096x4096/fp16"

# 客户问题复现
"customer_issues/github_issue_123/repro_case"
```

### 优点

- ✅ 易于过滤和查询
- ✅ 清晰的层次关系
- ✅ 便于批量操作
- ✅ 支持通配符匹配（如果实现）

---

## 🏗️ 架构设计

### GPU模式

#### Exclusive模式
- GPU一次只运行一个任务
- 适用于性能测试
- 保证测试结果的准确性

#### Shared模式
- GPU可以同时运行多个任务
- 受两个因素控制：
  - `max_concurrent_tasks`: 最大并发任务数（默认: 4）
  - `memory_threshold`: 内存使用阈值（默认: 75%）
- 适用于功能测试

### 任务驱动的GPU模式切换

#### 自动切换
- 任务开始时，根据 `task_mode` 自动设置GPU模式
- 任务结束时，自动恢复GPU模式

#### 手动覆盖
- 管理员可以设置手动模式（`manual_mode`）
- 手动模式优先级最高，持续到清除为止

#### 模式优先级
1. **手动模式** (`manual_mode`): 最高优先级
2. **任务锁定** (`mode_locked_by`): exclusive任务运行时
3. **默认模式**: shared

---

## 🔄 任务调度

### 调度策略

1. **全局队列**: 所有待处理任务
2. **GPU队列**: 每个GPU的专属队列
3. **FIFO策略**: 先进先出

### 调度流程

```
提交任务 → 全局队列 → 调度器分配GPU → GPU队列 → 任务执行 → 更新状态
                              ↓
                        GPU管理器监控
                   （显存、状态、错误检测）
```

### 错误处理

#### 严重错误处理
- 当检测到GPU严重错误时：
  1. 所有正在运行的任务被强制终止（SIGTERM → SIGKILL）
  2. 被终止的任务重新插入队列最前面
  3. 系统暂停调度 60 秒
  4. 自动恢复调度

#### 手动恢复
- 也可手动调用API立即恢复

---

## 🎨 设计优势

### 1. 概念清晰分离

- **task_mode**: 技术层面（GPU如何执行）
- **task_type**: 业务层面（测试分类）
- **task_label**: 标识层面（具体识别）

### 2. 灵活性

- 所有参数都是可选的
- 智能默认值减少输入
- 可以按需指定任何参数

### 3. 可扩展性

- `task_type` 可以添加更多类型
- `task_label` 支持任意字符串
- 不破坏现有代码

### 4. 易用性

```python
# 90%的情况
client.submit_task("test.py")

# 5%的情况（需要分类）
client.submit_task("test.py", task_type="functional")

# 3%的情况（需要标识）
client.submit_task("test.py", task_type="functional", task_label="...")

# 2%的情况（特殊需求）
client.submit_task("test.py", task_mode="exclusive", task_type="functional", task_label="...")
```

---

## 📦 数据模型

### Task Object

```python
{
  "task_id": "uuid",
  "task_mode": "shared",                              # 必有
  "task_type": "functional",                          # 可选（None时不显示）
  "task_label": "xpiler_cuda/add_3_3_256/cuda_vs_triton",  # 可选（None时不显示）
  "script_path": "/path/to/test.py",
  "status": "completed",
  "exit_code": 0,
  ...
}
```

### 条件序列化

- `task_type=None` → 响应中不包含 `task_type` 字段
- `task_label=None` → 响应中不包含 `task_label` 字段
- 减少数据传输，保持响应简洁

---

## 🚀 性能考虑

### GPU内存监控

- 实时监控GPU内存使用
- 任务分配前主动更新内存信息
- 确保准确的调度决策

### 并发控制

- Shared模式下限制并发任务数
- 防止GPU过载
- 平衡性能和资源利用

### 日志系统

- 双日志系统：会话日志 + 历史日志
- 任务日志单独存储（stdout/stderr）
- 支持大文件日志（不限制大小）

---

## 📚 相关文档

- [快速入门](QUICKSTART.md)
- [API参考手册](API_REFERENCE.md)
- [变更说明](CHANGES.md)

---

**最后更新:** 2025-01-15

