# NVGPU Server 快速入门

**版本:** v2.5  
**5分钟上手 GPU 任务调度服务器**

---

## 🚀 快速开始

### 1. 启动服务器

```bash
cd /workspace/server/nvgpu
python3 main.py
```

服务器将在 `http://localhost:8080` 启动。

### 2. (可选) 使用 GPU 配置

```bash
python3 main.py --gpu-config configs/gpu_resource.yml
```

---

## 💡 基础使用

### Python 客户端

```python
import sys
sys.path.insert(0, '/workspace/server/nvgpu')
from server.nvgpu.client import NVGPUClient

# 连接服务器
client = NVGPUClient("http://localhost:8080")

# ========== v2.5 核心特性 ==========

# 1. 最简用法（自动 shared 模式）
task_id = client.submit_task("test.py")

# 2. 功能测试（自动 shared 模式）
task_id = client.submit_task(
    "test.py",
    task_type="functional"
)

# 3. 性能测试（自动 exclusive 模式）
task_id = client.submit_task(
    "benchmark.py",
    task_type="performance"
)

# 4. 带具体标签
task_id = client.submit_task(
    "test_add.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)

# 5. 等待任务完成
import time
while True:
    result = client.get_task(task_id)
    if result.status in ['completed', 'failed', 'cancelled']:
        break
    time.sleep(1)

# 6. 检查结果
if result.exit_code == 0:
    print("✓ Task succeeded")
    stdout = client.get_full_task_log(task_id, "stdout")
else:
    print("✗ Task failed")
    stderr = client.get_full_task_log(task_id, "stderr")
```

---

## 🎯 核心概念

### 三个参数，清晰分离

#### 1. **task_mode** - GPU 行为控制
- **用途:** 控制 GPU 如何执行任务
- **取值:** `"exclusive"` 或 `"shared"`
- **默认:** 智能默认（基于 `task_type`）

#### 2. **task_type** - 业务分类
- **用途:** 业务层面的分类，用于统计和筛选
- **取值:** 
  - `"functional"` - 功能测试 → 自动 `shared` 模式
  - `"performance"` - 性能测试 → 自动 `exclusive` 模式
  - `"both"` - 两者兼有 → 自动 `exclusive` 模式（包含性能测试）
- **默认:** `None`（可选）

#### 3. **task_label** - 具体标识
- **用途:** 具体的测试标签，精确识别
- **取值:** 任意字符串（建议使用层次结构）
- **示例:**
  - `"xpiler_cuda/add_3_3_256/cuda_vs_triton"`
  - `"matmul/4096x4096/baseline"`
- **默认:** `None`（可选）

---

## 📊 使用场景

### 场景 1: 简单功能测试

```python
# 最简单（90% 的情况）
client.submit_task("test.py")

# 稍微详细
client.submit_task("test.py", task_type="functional")
```

### 场景 2: 带标签的功能测试

```python
client.submit_task(
    "test_add.py",
    task_type="functional",
    task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton"
)
```

### 场景 3: 性能基准测试

```python
# 自动使用 exclusive 模式
client.submit_task(
    "benchmark.py",
    task_type="performance",
    task_label="matmul/4096x4096/baseline"
)
```

### 场景 4: 特殊需求

```python
# 功能测试但需要独占 GPU（测试显存极限）
client.submit_task(
    "memory_stress.py",
    task_mode="exclusive",  # 显式覆盖
    task_type="functional",
    task_label="stress_test/memory_limit"
)
```

---

## 🔍 查询和过滤

### 查看 GPU 状态

```python
gpus = client.list_gpus()
for gpu in gpus:
    print(f"GPU {gpu['gpu_id']}: {gpu['status']}, "
          f"mode={gpu['mode']}, "
          f"tasks={gpu['running_task_count']}/{gpu['max_concurrent_tasks']}")
```

### 按类型过滤任务

```python
# 所有功能测试
functional_tasks = client.list_tasks(task_type="functional")

# 所有性能测试
perf_tasks = client.list_tasks(task_type="performance")

# 按状态过滤
running = client.list_tasks(status="running")
```

### 按标签过滤

```python
# 特定测试
cuda_tests = client.list_tasks(task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton")
```

---

## 🔧 GPU 管理

### 设置 GPU 模式（手动覆盖）

```python
# 设置手动模式（持续到清除）
client.set_gpu_mode(0, "exclusive", manual=True)

# 清除手动模式（恢复任务驱动）
client.clear_gpu_manual_mode(0)
```

### 配置 GPU 参数

```python
client.configure_gpu(
    0,
    memory_threshold=0.8,
    max_concurrent_tasks=6
)
```

---

## 📋 curl 示例

### 提交任务

```bash
# 最简单
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{"script_path": "/workspace/test.py"}'

# 带分类
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "script_path": "/workspace/test.py",
    "task_type": "functional"
  }'

# 完整示例
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "script_path": "/workspace/test.py",
    "task_type": "functional",
    "task_label": "xpiler_cuda/add_3_3_256/cuda_vs_triton",
    "args": ["--verbose"]
  }'
```

### 查询任务

```bash
# 查看任务状态
curl http://localhost:8080/tasks/{task_id}

# 列出所有任务
curl http://localhost:8080/tasks

# 过滤功能测试
curl "http://localhost:8080/tasks?task_type=functional"
```

### 查询 GPU

```bash
# 列出所有 GPU
curl http://localhost:8080/gpus

# 查看特定 GPU
curl http://localhost:8080/gpus/0
```

---

## 🎨 项目结构

```
/workspace/server/nvgpu/
├── main.py              # 主入口
├── config.py            # 服务器配置
├── models.py            # 数据模型（GPU、Task）
├── logger.py            # 统一日志
├── gpu_manager.py       # GPU 管理器
├── task_queue.py        # 任务队列
├── task_runner.py       # 任务执行器
├── scheduler.py         # 任务调度器
├── api_server.py        # REST API
├── client.py            # Python 客户端
├── gpu_config_loader.py # GPU 配置加载
├── configs/
│   └── gpu_resource.yml # GPU 配置文件
├── examples/            # 示例脚本
├── test_scripts/        # 测试脚本
└── docs/                # 文档
```

---

## 📚 深入学习

### 核心文档

- **[v2.5 最终设计](V2.5_FINAL_DESIGN.md)** ⭐⭐⭐⭐ - 完整的设计说明
- **[API 参考手册（中文）](API_REFERENCE_ZH.md)** ⭐⭐⭐ - 完整的 API 文档
- **[v2.5 变更说明](V2.5_CHANGES.md)** - 版本变更详情

### 配置参考

- **[GPU 配置指南](GPU_CONFIG_GUIDE.md)** - 如何配置 GPU 资源
- **[工作目录指南](WORK_DIR_GUIDE.md)** - work_dir 功能说明
- **[目录结构](DIRECTORY_STRUCTURE.md)** - 文件组织说明

### 架构文档

- **[v2.0 架构说明（中文）](ARCHITECTURE_V2_ZH.md)** - 任务驱动模式系统
- **[v2.0 架构说明（英文）](ARCHITECTURE_V2.md)** - Task-Driven Mode System

---

## 💡 最佳实践

### 标签命名

推荐使用层次结构：

```
<项目>/<测试名称>/<子分类>
```

示例：
```python
"xpiler_cuda/add_3_3_256/cuda_vs_triton"
"benchmark/matmul/4096x4096/fp32"
"nightly_regression/test_suite_1/case_001"
```

### 错误处理

```python
try:
    task_id = client.submit_task("test.py")
    result = wait_for_task(client, task_id)
    
    if result.exit_code != 0:
        stderr = client.get_full_task_log(task_id, "stderr")
        raise RuntimeError(f"Task failed: {stderr}")
        
except TimeoutError:
    client.cancel_task(task_id)
    raise
```

---

## 🚨 常见问题

### Q: 如何查看服务器日志？

A: 日志位于 `logs/` 目录：
- `nvgpu_server.log` - 历史日志（追加）
- `nvgpu_server_YYYYMMDD_HHMMSS.log` - 当前会话日志

### Q: 任务失败了怎么办？

A: 使用客户端获取日志：
```python
stderr = client.get_full_task_log(task_id, "stderr")
stdout = client.get_full_task_log(task_id, "stdout")
```

### Q: 如何取消正在运行的任务？

A: 使用取消接口：
```python
client.cancel_task(task_id)
```

### Q: GPU 出现错误怎么办？

A: 服务器会自动暂停 60 秒后恢复。也可手动清除：
```python
client.clear_severe_error()
```

---

## 🎓 完整示例

### 批量功能测试

```python
#!/usr/bin/env python3
"""批量功能测试示例"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')
from server.nvgpu.client import NVGPUClient
import time

client = NVGPUClient("http://localhost:8080")

# 定义测试列表
tests = [
    ("test1.py", "xpiler_cuda/test1"),
    ("test2.py", "xpiler_cuda/test2"),
    ("test3.py", "xpiler_cuda/test3"),
]

# 提交所有测试
task_ids = []
for script, label in tests:
    task_id = client.submit_task(
        script,
        task_type="functional",
        task_label=label
    )
    task_ids.append((label, task_id))
    print(f"Submitted: {label} - {task_id[:8]}")

# 等待所有完成
results = []
for label, task_id in task_ids:
    while True:
        result = client.get_task(task_id)
        if result.status in ['completed', 'failed']:
            break
        time.sleep(1)
    
    results.append((label, result))
    print(f"{label}: {result.status}")

# 统计结果
success = sum(1 for _, r in results if r.exit_code == 0)
print(f"\n✓ {success}/{len(results)} tests passed")
```

---

**版本:** v2.5  
**更新时间:** 2025-01-15  
**完整文档:** [docs/README.md](README.md)
