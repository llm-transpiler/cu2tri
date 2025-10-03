# NVGPU Client Examples

Python客户端使用示例集合。

## 前置条件

确保NVGPU服务器正在运行：

```bash
cd /workspace/server/nvgpu
python main.py --gpus 0 1
```

## 示例列表

### 1. `example_basic.py` - 基础使用

最简单的示例：提交任务并等待完成。

```bash
python examples/example_basic.py
```

**演示内容：**
- 健康检查
- 提交简单任务
- 等待任务完成
- 检查结果

---

### 2. `example_batch_submit.py` - 批量提交

批量提交多个任务并监控进度。

```bash
python examples/example_batch_submit.py
```

**演示内容：**
- 批量提交不同类型的任务
- 实时监控所有任务状态
- 汇总统计结果

**输出示例：**
```
Progress: 2/4 completed
  Functional Test 1: ✓ Completed (exit=0)
  Memory Test: Running...
  Performance Benchmark: Queued
  Long Task: ✓ Completed (exit=0)
```

---

### 3. `example_gpu_management.py` - GPU管理

动态管理GPU设置。

```bash
python examples/example_gpu_management.py
```

**演示内容：**
- 列出所有GPU及其状态
- 切换GPU模式（exclusive/shared）
- 修改内存阈值
- 指定GPU运行任务
- 查看服务器统计信息

---

### 4. `example_error_handling.py` - 错误处理

展示各种错误场景的处理。

```bash
python examples/example_error_handling.py
```

**演示内容：**
- 成功任务处理
- 异常处理
- 非零退出码处理
- 任务取消
- 超时处理

---

### 5. `example_custom_env.py` - 自定义环境变量

向任务传递自定义环境变量。

```bash
python examples/example_custom_env.py
```

**演示内容：**
- 设置自定义环境变量
- 在任务中读取环境变量
- 验证环境变量正确传递

---

## 在你自己的项目中使用

### 安装依赖

```bash
pip install requests
```

### 导入客户端

```python
from nvgpu.client import NVGPUClient

# 创建客户端
client = NVGPUClient("http://localhost:8080")

# 提交任务
task_id = client.submit_task(
    script_path="/path/to/your_script.py",
    task_type="functional",
    args=["--your-arg", "value"]
)

# 等待完成
result = client.wait_for_task(task_id)
print(f"Status: {result.status}, Exit code: {result.exit_code}")
```

### 常用模式

#### 模式1：提交并等待

```python
client = NVGPUClient()
task_id = client.submit_task(script_path="test.py", task_type="functional")
result = client.wait_for_task(task_id, timeout=600)

if result.status == "completed" and result.exit_code == 0:
    print("Success!")
else:
    print(f"Failed: {result.error_message}")
```

#### 模式2：批量提交

```python
client = NVGPUClient()
task_ids = []

for script in ["test1.py", "test2.py", "test3.py"]:
    task_id = client.submit_task(script_path=script, task_type="functional")
    task_ids.append(task_id)

# 等待所有任务
results = [client.wait_for_task(tid) for tid in task_ids]
```

#### 模式3：异步监控

```python
import time

client = NVGPUClient()
task_id = client.submit_task(script_path="long_task.py", task_type="performance")

while True:
    result = client.get_task(task_id)
    if result.status in ["completed", "failed", "cancelled"]:
        break
    print(f"Status: {result.status}")
    time.sleep(5)
```

#### 模式4：指定GPU

```python
client = NVGPUClient()

# 在GPU 0上运行
task_id = client.submit_task(
    script_path="test.py",
    task_type="functional",
    gpu_id=0
)
```

#### 模式5：GPU管理

```python
client = NVGPUClient()

# 列出所有GPU
gpus = client.list_gpus()
for gpu in gpus:
    print(f"GPU {gpu['gpu_id']}: {gpu['status']}")

# 设置GPU为独占模式
client.set_gpu_mode(gpu_id=0, mode="exclusive")

# GPU下线维护
client.set_gpu_status(gpu_id=1, status="offline")
```

## API参考

完整的客户端API文档：

### 任务管理

- `submit_task()` - 提交任务
- `get_task()` - 获取任务信息
- `wait_for_task()` - 等待任务完成
- `cancel_task()` - 取消任务
- `list_tasks()` - 列出所有任务

### GPU管理

- `list_gpus()` - 列出所有GPU
- `get_gpu()` - 获取GPU信息
- `set_gpu_status()` - 设置GPU状态
- `set_gpu_mode()` - 设置GPU模式
- `set_gpu_memory_threshold()` - 设置内存阈值
- `register_gpu()` - 注册GPU
- `unregister_gpu()` - 注销GPU

### 监控

- `health_check()` - 健康检查
- `get_stats()` - 获取统计信息

## 测试脚本

可用的测试脚本位于 `test_scripts/` 目录：

- `simple_functional_test.py` - 简单功能测试
- `memory_stress_test.py` - 内存压力测试
- `performance_benchmark.py` - 性能基准测试
- `long_running_task.py` - 长时间运行任务
- `multi_gpu_test.py` - 多GPU测试
- `failing_test.py` - 故意失败的测试（用于测试错误处理）

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
    print(f"Log: {result.log_file}")
    # 读取日志文件查看详细信息
```

### 超时

```python
try:
    result = client.wait_for_task(task_id, timeout=300)
except TimeoutError:
    print("Task timed out")
    client.cancel_task(task_id)
```

