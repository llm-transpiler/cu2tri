# NVGPU 示例集合

所有客户端使用示例，演示 NVGPU 服务器的各项功能。

## 快速开始

**1. 启动服务器：**
```bash
cd /workspace/server/nvgpu
python main.py
```

**2. 运行示例（新终端）：**
```bash
cd /workspace/server/nvgpu
python examples/example_basic.py
```

## 示例列表

### 基础功能

#### 📝 `example_basic.py` - 基础使用
最简单的示例，提交任务并等待完成。
```bash
python examples/example_basic.py
```
**演示：** 健康检查、提交任务、等待完成、检查结果

---

#### ⏱️ `example_task_timing.py` - 任务计时分析 ⭐ 新增
分析任务的详细时间开销，了解性能瓶颈。
```bash
python examples/example_task_timing.py
```
**演示：**
- 详细时间字段（pending, queue, waiting, execution, total）
- 时间线可视化
- 性能瓶颈分析
- 执行效率计算

**时间字段说明（毫秒，2位小数）：**
- `pending_time_ms`: 提交→分配GPU
- `queue_time_ms`: 分配GPU→开始执行
- `waiting_time_ms`: 总等待时间
- `execution_time_ms`: 实际执行时间
- `total_time_ms`: 端到端总时间

---

#### 📦 `example_batch_submit.py` - 批量提交
批量提交多个任务并监控进度。
```bash
python examples/example_batch_submit.py
```
**演示：** 批量提交、实时监控、汇总统计

---

#### 🔄 `example_concurrent_tasks.py` - 并发任务
多任务并发执行。
```bash
python examples/example_concurrent_tasks.py
```
**演示：** 并发提交、同时运行多个任务、性能对比

---

### 任务参数测试 ⭐ 新增

#### 🧪 `example_task_parameters.py` - 完整参数测试
验证 `task_type`、`task_mode`、`task_label` 所有参数组合（15个测试用例）。
```bash
python examples/example_task_parameters.py
```
**测试：**
- ✓ 参数可选性（无参数、单参数、多参数）
- ✓ 智能默认值（functional→shared, performance→exclusive, both→exclusive）
- ✓ 显式覆盖（task_mode 覆盖 task_type 默认值）
- ✓ 所有参数组合正常工作

**运行时间：** ~7-10分钟

---

#### ⚡ `example_task_parameters_quick.py` - 快速参数测试
核心参数组合测试（12个测试用例），适合快速验证。
```bash
python examples/example_task_parameters_quick.py
```
**运行时间：** ~5-7分钟

**测试矩阵：**
```
参数组合            | task_type    | task_mode   | task_label | 预期模式
-------------------|--------------|-------------|------------|----------
无参数             | -            | -           | -          | shared
仅 mode           | -            | shared      | -          | shared
仅 mode           | -            | exclusive   | -          | exclusive
仅 type           | functional   | -           | -          | shared
仅 type           | performance  | -           | -          | exclusive
仅 type           | both         | -           | -          | exclusive
覆盖智能默认        | functional   | exclusive   | -          | exclusive
覆盖智能默认        | performance  | shared      | -          | shared
```

---

### 任务管理

#### ❌ `example_cancel_running_task.py` - 任务取消
取消 pending/queued/running 状态的任务。
```bash
python examples/example_cancel_running_task.py
```
**演示：**
- 取消排队任务（force=False）
- 强制取消运行中任务（force=True）
- 批量取消多个任务
- 错误处理（不存在的任务）

---

#### 📄 `example_log_handling.py` - 日志管理
读取和管理任务日志。
```bash
python examples/example_log_handling.py
```
**演示：**
- 读取任务摘要日志
- 读取 stdout/stderr
- 处理大日志文件（分段读取）
- 下载日志到本地

---

#### ⚠️ `example_error_handling.py` - 错误处理
各种错误场景的处理。
```bash
python examples/example_error_handling.py
```
**演示：**
- 正常完成的任务
- 脚本异常（Python 异常）
- 非零退出码
- 超时处理
- 任务取消

---

### GPU 管理

#### 🎮 `example_gpu_management.py` - GPU 管理
动态管理 GPU 设置。
```bash
python examples/example_gpu_management.py
```
**演示：**
- 列出所有 GPU 及状态
- 切换 GPU 模式（exclusive/shared）
- 修改内存阈值
- 设置最大并发任务数
- 指定 GPU 运行任务
- 查看服务器统计信息

---

#### 🔀 `example_mode_switching.py` - GPU 模式切换
GPU 模式动态切换及对运行中任务的影响。
```bash
python examples/example_mode_switching.py
```
**演示：**
- Shared → Exclusive 切换（运行中任务不受影响）
- Exclusive 模式行为（同时最多1个任务）
- Shared 模式并发控制（max_concurrent_tasks）

**关键特性：**
- ✓ 模式切换不中断运行中任务（优雅过渡）
- ✓ 新任务遵守新模式约束
- ✓ 安全的动态资源管理

---

### 高级功能

#### 🌍 `example_custom_env.py` - 自定义环境变量
向任务传递自定义环境变量。
```bash
python examples/example_custom_env.py
```
**演示：**
- 设置自定义环境变量
- 在任务中读取环境变量
- 验证环境变量正确传递

---

## 使用客户端 API

### 安装
```bash
pip install requests
```

### 基本用法
```python
from nvgpu.client import NVGPUClient

# 创建客户端
client = NVGPUClient("http://localhost:8080")

# 提交任务
task_id = client.submit_task(
    script_path="/path/to/script.py",
    task_type="functional",  # 可选: functional/performance/both
    task_mode="shared",      # 可选: shared/exclusive (覆盖智能默认)
    task_label="my_test",    # 可选: 自定义标签
    args=["--arg", "value"]
)

# 等待完成
result = client.wait_for_task(task_id, timeout=600)

if result.status == "completed" and result.exit_code == 0:
    print("✓ Success!")
else:
    print(f"✗ Failed: {result.error_message}")
```

### 常用模式

**模式 1：提交并等待**
```python
task_id = client.submit_task(script_path="test.py", task_type="functional")
result = client.wait_for_task(task_id)
```

**模式 2：批量提交**
```python
task_ids = []
for script in ["test1.py", "test2.py", "test3.py"]:
    task_id = client.submit_task(script_path=script, task_type="functional")
    task_ids.append(task_id)

results = [client.wait_for_task(tid) for tid in task_ids]
```

**模式 3：异步监控**
```python
import time

task_id = client.submit_task(script_path="long_task.py", task_type="performance")

while True:
    result = client.get_task(task_id)
    if result.status in ["completed", "failed", "cancelled"]:
        break
    print(f"Status: {result.status}")
    time.sleep(5)
```

**模式 4：指定 GPU**
```python
task_id = client.submit_task(
    script_path="test.py",
    task_type="functional",
    gpu_id=0  # 在 GPU 0 上运行
)
```

**模式 5：显式控制模式**
```python
# 功能测试但需要独占 GPU
task_id = client.submit_task(
    script_path="test.py",
    task_type="functional",
    task_mode="exclusive"  # 覆盖 functional 的默认 shared 模式
)
```

### 智能默认值

客户端会根据 `task_type` 自动选择合适的 `task_mode`：

| task_type | 默认 task_mode | 原因 |
|-----------|---------------|------|
| `functional` | `shared` | 功能测试可共享 GPU |
| `performance` | `exclusive` | 性能测试需独占 GPU |
| `both` | `exclusive` | 保守选择 |
| 未指定 | `shared` | 默认值 |

**显式指定 `task_mode` 可覆盖智能默认值。**

### 客户端 API 参考

**任务管理：**
- `submit_task()` - 提交任务
- `get_task()` - 获取任务信息
- `wait_for_task()` - 等待任务完成
- `cancel_task()` - 取消任务
- `list_tasks()` - 列出所有任务
- `get_task_log()` - 获取任务日志

**GPU 管理：**
- `list_gpus()` - 列出所有 GPU
- `get_gpu()` - 获取 GPU 信息
- `set_gpu_status()` - 设置 GPU 状态（online/offline/maintenance）
- `set_gpu_mode()` - 设置 GPU 模式（exclusive/shared）
- `set_gpu_memory_threshold()` - 设置内存阈值
- `set_gpu_max_concurrent_tasks()` - 设置最大并发任务数
- `register_gpu()` - 注册 GPU
- `unregister_gpu()` - 注销 GPU

**监控：**
- `health_check()` - 健康检查
- `get_stats()` - 获取统计信息

完整 API 文档请参考 `docs/API_REFERENCE_ZH.md`。

## 测试脚本

可用的测试脚本（位于 `test_scripts/` 目录）：

- `simple_functional_test.py` - 简单功能测试（几秒钟）
- `memory_stress_test.py` - 内存压力测试
- `performance_benchmark.py` - 性能基准测试
- `long_running_task.py` - 长时间运行任务（可配置时长）
- `multi_gpu_test.py` - 多 GPU 测试
- `failing_test.py` - 故意失败的测试（用于测试错误处理）

## 推荐运行顺序

**初次使用：**
1. `example_basic.py` - 了解基础流程
2. `example_task_parameters_quick.py` - 验证参数功能
3. `example_gpu_management.py` - 学习 GPU 管理
4. `example_batch_submit.py` - 批量任务处理

**深入学习：**
5. `example_mode_switching.py` - GPU 模式切换
6. `example_cancel_running_task.py` - 任务取消
7. `example_error_handling.py` - 错误处理
8. `example_log_handling.py` - 日志管理

**全面测试：**
9. `example_task_parameters.py` - 完整参数测试（15个测试用例）

## 故障排除

### 连接失败
```python
if not client.health_check():
    print("Server not responding. Is it running?")
```

### 任务失败
```python
result = client.get_task(task_id)
if result.status == "failed":
    print(f"Error: {result.error_message}")
    # 获取详细日志
    log = client.get_task_log(task_id, log_type="summary")
    print(log["content"])
```

### 任务超时
```python
try:
    result = client.wait_for_task(task_id, timeout=300)
except TimeoutError:
    print("Task timed out")
    client.cancel_task(task_id, force=True)
```

### 参数测试失败
如果 `example_task_parameters*.py` 测试失败：
1. 检查服务器日志：`tail -f logs/nvgpu_server.log`
2. 确认 GPU 可用：`curl http://localhost:8080/gpus`
3. 验证测试脚本存在：`ls -la test_scripts/simple_functional_test.py`

## 更多资源

- **快速入门：** `docs/QUICKSTART.md`
- **API 参考：** `docs/API_REFERENCE_ZH.md`
- **项目文档：** `docs/README.md`
- **配置指南：** `configs/gpu_resource.yml`

## 技术支持

遇到问题？
1. 查看服务器日志：`logs/nvgpu_server.log`
2. 查看任务日志：`logs/tasks/<task_id>.log`
3. 检查 GPU 状态：`python -c "from nvgpu.client import NVGPUClient; print(NVGPUClient().list_gpus())"`
