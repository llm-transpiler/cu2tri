# NVGPU Server 项目总结

## 📋 项目概览

一个专业的 GPU 任务调度服务器，用于管理和执行基于 Python 的 GPU 工作负载。

**核心特性：**
- ✅ GPU 独占/共享模式控制
- ✅ 实时显存监控和阈值管理
- ✅ 严重错误自动暂停和恢复
- ✅ FIFO 任务调度
- ✅ RESTful API
- ✅ 完整的日志系统
- ✅ Python 客户端库

---

## 📁 项目结构

```
/workspace/server/nvgpu/
├── 核心模块
│   ├── config.py              # 配置管理（模式、类型、服务器设置）
│   ├── models.py              # 数据模型（GPU、Task）
│   ├── logger.py              # 统一日志配置
│   ├── gpu_manager.py         # GPU 管理器（监控、状态、错误处理）
│   ├── task_queue.py          # 任务队列（全局队列 + GPU队列）
│   ├── task_runner.py         # 任务执行器（subprocess）
│   ├── scheduler.py           # 任务调度器（FIFO）
│   ├── api_server.py          # REST API 服务器（Flask）
│   └── main.py                # 主入口程序
│
├── 客户端
│   ├── client.py              # Python 客户端库
│   └── examples/              # 使用示例
│       ├── README.md          # 客户端文档
│       ├── example_basic.py   # 基础示例
│       ├── example_batch_submit.py      # 批量提交
│       ├── example_gpu_management.py    # GPU管理
│       ├── example_error_handling.py    # 错误处理
│       └── example_custom_env.py        # 自定义环境
│
├── 测试脚本
│   └── test_scripts/
│       ├── simple_functional_test.py    # 简单功能测试
│       ├── memory_stress_test.py        # 内存压力测试
│       ├── performance_benchmark.py     # 性能基准测试
│       ├── long_running_task.py         # 长时间运行任务
│       ├── multi_gpu_test.py            # 多GPU测试
│       └── failing_test.py              # 失败测试（测试错误处理）
│
├── 文档
│   ├── README.md              # 完整文档（英文）
│   ├── QUICKSTART.md          # 快速开始（中文）
│   └── PROJECT_SUMMARY.md     # 项目总结（本文件）
│
└── 其他
    ├── requirements.txt       # Python 依赖
    ├── test_api.sh           # API 测试脚本
    ├── __init__.py           # 包初始化
    └── .gitignore            # Git 忽略配置
```

**总计：** 9个核心模块 + 1个客户端 + 5个示例 + 6个测试脚本 + 3个文档

---

## 🎯 核心设计

### 1. GPU 模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `exclusive` | 独占模式，每次只运行一个任务 | 显存密集型任务 |
| `shared` | 共享模式，多任务并行（显存阈值控制） | 高吞吐量场景 |

### 2. 任务类型

| 类型 | 说明 |
|------|------|
| `functional` | 功能验证和测试 |
| `performance` | 性能测试和基准测试 |

### 3. 架构流程

```
┌─────────┐
│ 用户提交 │
└────┬────┘
     ↓
┌─────────────┐
│ 全局任务队列 │ (pending)
└──────┬──────┘
       ↓
┌──────────┐
│  调度器   │ ←──────┐
└─────┬────┘         │
      ↓              │
┌─────────────┐      │
│ GPU任务队列  │      │ GPU管理器
│  (每GPU)    │      │ (监控/状态)
└──────┬──────┘      │
       ↓             │
┌──────────┐         │
│ 任务执行  │─────────┘
└──────────┘
```

### 4. 严重错误处理

- 检测到 GPU 错误 → 自动暂停所有调度
- 已运行任务继续执行
- 需要手动调用 `/gpus/clear_error` 恢复

---

## 🚀 快速开始

### 1. 安装依赖
```bash
cd /workspace/server/nvgpu
pip install -r requirements.txt
```

### 2. 启动服务器
```bash
python main.py --gpus 0 1 2 3
```

### 3. 使用 Python 客户端
```python
from nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")
task_id = client.submit_task(
    script_path="/path/to/test.py",
    task_type="functional"
)
result = client.wait_for_task(task_id)
print(f"Status: {result.status}")
```

### 4. 或使用 REST API
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "/path/to/test.py",
    "args": ["--arg1", "value"]
  }'
```

---

## 🔧 配置说明

### 命令行参数

```bash
python main.py \
  --gpus 0 1 2 3              # 启动时注册的 GPU
  --gpu-mode shared           # 默认模式 (exclusive/shared)
  --memory-threshold 0.75     # 显存阈值 (0-1)
  --port 8080                 # API 端口
  --host 0.0.0.0             # 监听地址
  --log-level INFO           # 日志级别
  --log-file logs/server.log # 日志文件
```

### 配置文件 (config.py)

```python
class ServerConfig:
    default_gpu_mode = GPUMode.SHARED
    default_memory_threshold = 0.75
    task_timeout = 600              # 任务超时（秒）
    error_pause_duration = 300      # 错误暂停时长（秒）
    scheduler_interval = 1.0        # 调度间隔（秒）
    gpu_monitor_interval = 5.0      # 监控间隔（秒）
```

---

## 📡 REST API 端点

### GPU 管理

```
GET    /gpus                         # 列出所有 GPU
GET    /gpus/{gpu_id}                # 获取 GPU 详情
PUT    /gpus/{gpu_id}/status         # 设置状态 (online/offline/maintenance)
PUT    /gpus/{gpu_id}/mode           # 设置模式 (exclusive/shared)
PUT    /gpus/{gpu_id}/memory_threshold  # 设置显存阈值
POST   /gpus/register                # 注册 GPU
POST   /gpus/{gpu_id}/unregister     # 注销 GPU
POST   /gpus/{gpu_id}/error          # 触发严重错误
POST   /gpus/clear_error             # 清除错误状态
```

### 任务管理

```
POST   /tasks                        # 提交任务
GET    /tasks                        # 列出所有任务
GET    /tasks/{task_id}              # 获取任务详情
POST   /tasks/{task_id}/cancel       # 取消任务
```

### 监控

```
GET    /health                       # 健康检查
GET    /stats                        # 服务器统计
```

---

## 💡 使用场景

### 场景 1：简单功能测试

```python
client = NVGPUClient()
task_id = client.submit_task(
    script_path="test.py",
    task_type="functional"
)
result = client.wait_for_task(task_id)
```

### 场景 2：批量任务提交

```python
client = NVGPUClient()
task_ids = []
for script in test_scripts:
    tid = client.submit_task(script_path=script, task_type="functional")
    task_ids.append(tid)

# 等待所有完成
results = [client.wait_for_task(tid) for tid in task_ids]
```

### 场景 3：指定 GPU 运行

```python
client = NVGPUClient()
task_id = client.submit_task(
    script_path="test.py",
    task_type="performance",
    gpu_id=0  # 指定在 GPU 0 上运行
)
```

### 场景 4：GPU 维护

```python
client = NVGPUClient()

# 下线 GPU 1 进行维护
client.set_gpu_status(gpu_id=1, status="offline")

# 维护完成后上线
client.set_gpu_status(gpu_id=1, status="online")
```

### 场景 5：动态调整 GPU 模式

```python
client = NVGPUClient()

# 显存密集任务：改为独占模式
client.set_gpu_mode(gpu_id=0, mode="exclusive")
task_id = client.submit_task(script_path="heavy_task.py", gpu_id=0)

# 任务完成后恢复共享模式
client.wait_for_task(task_id)
client.set_gpu_mode(gpu_id=0, mode="shared")
```

---

## 📊 测试脚本说明

### 1. simple_functional_test.py
- 快速 GPU 可用性检查
- 简单矩阵运算验证
- 适合：冒烟测试

### 2. memory_stress_test.py
- 内存分配压力测试
- 支持指定分配大小和迭代次数
- 适合：测试显存管理

### 3. performance_benchmark.py
- 矩阵乘法性能测试（TFLOPS）
- 内存带宽测试（H2D/D2H）
- 适合：性能基准测试

### 4. long_running_task.py
- 长时间运行任务模拟
- 定期进度报告
- 适合：测试超时和取消

### 5. multi_gpu_test.py
- 验证 GPU 分配正确性
- 检查 CUDA_VISIBLE_DEVICES
- 适合：多 GPU 环境测试

### 6. failing_test.py
- 模拟各种失败场景
- 支持：异常、退出码、CUDA错误、超时
- 适合：测试错误处理

---

## 🔍 监控和日志

### 日志位置

- **服务器日志:** `logs/nvgpu_server.log`
- **任务日志:** `logs/tasks/{task_id}.log`

### 日志格式

```
2025-10-03 12:00:00 | gpu_manager | INFO | GPU 0 registered
2025-10-03 12:00:05 | scheduler | INFO | Task abc-123 queued for GPU 0
2025-10-03 12:00:06 | task_runner | INFO | Task abc-123 completed
```

### 监控统计

```python
stats = client.get_stats()
print(f"Pending: {stats['queue']['pending']}")
print(f"Running: {stats['queue']['running']}")
print(f"Online GPUs: {stats['gpus']['online_gpus']}")
```

---

## ✅ 最佳实践

1. **正常使用**
   - 使用 `shared` 模式 + 75% 阈值
   - 定期监控 `/stats` 端点

2. **显存密集任务**
   - 使用 `exclusive` 模式
   - 或提高 `shared` 模式的阈值

3. **GPU 维护**
   - 先设置 `offline` 状态
   - 等待运行任务完成
   - 进行硬件维护

4. **错误处理**
   - 检查任务日志文件
   - 分析错误原因
   - 必要时清除严重错误状态

5. **批量任务**
   - 先提交所有任务
   - 异步监控状态
   - 避免阻塞等待

---

## 🛠️ 技术栈

- **语言:** Python 3.8+
- **Web框架:** Flask
- **GPU监控:** pynvml (NVIDIA Management Library)
- **HTTP客户端:** requests
- **并发:** threading (线程安全队列)
- **日志:** logging (统一日志系统)

---

## 📦 依赖

```
flask>=2.3.0       # REST API 服务器
pynvml>=11.5.0     # GPU 监控
requests           # HTTP 客户端（仅客户端需要）
torch              # 测试脚本使用（可选）
```

---

## 🎓 学习路径

1. **初学者:** 从 `example_basic.py` 开始
2. **进阶:** 学习 `example_batch_submit.py` 和 `example_gpu_management.py`
3. **专家:** 研究错误处理和自定义环境变量
4. **架构理解:** 阅读核心模块源码

---

## 📈 性能特点

- **调度延迟:** ~1秒（可配置）
- **监控间隔:** ~5秒（可配置）
- **任务超时:** 600秒（可配置）
- **并发支持:** 每个 GPU 可同时运行多个任务（shared 模式）
- **线程安全:** 所有共享资源使用 RLock 保护

---

## 🔒 安全考虑

- ⚠️ 默认监听 0.0.0.0，建议生产环境使用反向代理
- ⚠️ 无身份认证，建议添加 API 认证层
- ⚠️ 任务可执行任意 Python 脚本，需要信任环境
- ✅ 自动设置 CUDA_VISIBLE_DEVICES 隔离 GPU
- ✅ 完整的错误捕获和日志记录

---

## 🚧 未来扩展

可能的扩展方向：

1. **优先级队列:** 支持任务优先级
2. **依赖管理:** 任务间依赖关系
3. **资源预留:** 提前预留 GPU 资源
4. **多机分布式:** 跨节点任务调度
5. **Web UI:** 图形化管理界面
6. **认证授权:** JWT/OAuth2 支持
7. **指标收集:** Prometheus/Grafana 集成
8. **数据库持久化:** 任务历史存储

---

## 📞 支持

- 文档: `README.md` (英文), `QUICKSTART.md` (中文)
- 示例: `examples/` 目录
- 测试: `test_scripts/` 目录

---

## 📄 许可证

内部使用

---

**版本:** 1.0.0  
**创建日期:** 2025-10-03  
**作者:** AI Assistant  
**状态:** ✅ 生产就绪

