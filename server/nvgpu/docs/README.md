# NVGPU Server

一个专业的 GPU 任务调度服务器，用于管理和执行基于 Python 的 GPU 工作负载。

## 特性

- **GPU 管理**: 支持多个 GPU，每个 GPU 可独立配置
- **两种 GPU 模式**:
  - `exclusive`: 每次只能运行一个任务（独占访问）
  - `shared`: 多个并发任务，双重控制：
    - 最大并发任务数限制（默认：3）
    - 显存阈值限制（默认：75%）
- **任务类型**:
  - `functional`: 功能验证和测试
  - `performance`: 完整性能测试和基准测试
- **严重错误处理**: GPU 错误时自动暂停并延迟处理
- **任务调度**: 经典操作系统风格的 FIFO 调度，每个 GPU 有独立队列
- **GPU 控制**: 通过 API 动态管理在线/离线状态
- **显存监控**: 实时 GPU 显存使用率跟踪
- **完整日志**: 所有组件的统一日志系统

## 架构

```
全局任务队列 → 调度器 → GPU 队列 → 任务执行
                  ↓
             GPU 管理器（监控显存、状态）
```

## 安装

```bash
cd /workspace/server/nvgpu
pip install -r requirements.txt
```

## 快速开始

### 1. 启动服务器

```bash
# 使用 GPU 配置文件启动（推荐）
python main.py

# 或手动指定 GPU
python main.py --gpus 0 1 2 3

# 使用自定义 GPU 配置文件
python main.py --gpu-config /path/to/gpu_resource.yml

# 使用自定义设置
python main.py --gpus 0 1 --gpu-mode shared --memory-threshold 0.8 --port 8080
```

### 2. 提交任务

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "/path/to/test_script.py",
    "work_dir": "/path/to/work/dir",
    "args": ["--arg1", "value1"],
    "gpu_id": 0
  }'
```

### 3. 检查任务状态

```bash
curl http://localhost:8080/tasks/{task_id}
```

## API 参考

### GPU 管理

- `GET /gpus` - 列出所有 GPU
- `GET /gpus/{gpu_id}` - 获取 GPU 详情
- `PUT /gpus/{gpu_id}/status` - 设置 GPU 状态（online/offline/maintenance）
- `PUT /gpus/{gpu_id}/mode` - 设置 GPU 模式（exclusive/shared）
- `PUT /gpus/{gpu_id}/memory_threshold` - 设置显存阈值
- `PUT /gpus/{gpu_id}/max_concurrent_tasks` - 设置最大并发任务数
- `POST /gpus/register` - 注册新 GPU
- `POST /gpus/{gpu_id}/unregister` - 注销 GPU
- `POST /gpus/{gpu_id}/error` - 触发严重错误（测试用）
- `POST /gpus/clear_error` - 清除严重错误状态

### 任务管理

- `POST /tasks` - 提交新任务
- `GET /tasks` - 列出所有任务（可选 `?status=pending`）
- `GET /tasks/{task_id}` - 获取任务详情
- `POST /tasks/{task_id}/cancel` - 取消待处理/排队的任务

### 监控

- `GET /stats` - 获取服务器统计信息
- `GET /health` - 健康检查

## 配置

### GPU 配置 (gpu_resource.yml)

定义可用的 GPU 及其映射：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "NVIDIA RTX 6000 Ada Generation"
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3  # shared 模式下的最大并发任务数
```

详细配置指南请参阅 [GPU_CONFIG_GUIDE.md](GPU_CONFIG_GUIDE.md)。  
并发任务控制详情请参阅 [MAX_CONCURRENT_TASKS.md](MAX_CONCURRENT_TASKS.md)。

### 服务器配置

编辑 `config.py` 或使用命令行参数：

```python
ServerConfig(
    host="0.0.0.0",
    port=8080,
    default_gpu_mode=GPUMode.SHARED,
    default_memory_threshold=0.75,
    default_max_concurrent_tasks=3,  # shared 模式下的最大并发任务数
    task_timeout=600,  # 秒
    error_pause_duration=60,  # 1 分钟，自动恢复
    scheduler_interval=1.0,  # 秒
    gpu_monitor_interval=5.0,  # 秒
)
```

## 任务执行

任务作为 Python 子进程执行，具有以下特性：
- 自动设置 `CUDA_VISIBLE_DEVICES`
- 捕获标准输出/标准错误
- 在 `logs/tasks/` 中生成日志文件
- 超时保护
- 自定义参数原样传递给脚本

## 示例

### 更改 GPU 模式

```bash
curl -X PUT http://localhost:8080/gpus/0/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "exclusive"}'
```

### 设置最大并发任务数

```bash
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 6}'
```

### 设置为离线维护

```bash
curl -X PUT http://localhost:8080/gpus/0/status \
  -H "Content-Type: application/json" \
  -d '{"status": "offline"}'
```

### 提交性能测试

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "performance",
    "script_path": "/path/to/benchmark.py",
    "work_dir": "/path/to/work",
    "args": ["--iterations", "1000"]
  }'
```

### 查看统计信息

```bash
curl http://localhost:8080/stats
```

## 日志

日志写入到：
- 服务器日志：`logs/nvgpu_server.log`
- 任务日志：`logs/tasks/{task_id}.log`

日志格式：`timestamp | component | level | message`

## 错误处理

当检测到严重的 GPU 错误时：
1. GPU 状态变为 `ERROR`
2. 所有任务调度暂停
3. 现有任务继续完成
4. **60 秒后自动恢复**（可配置）
5. 也可通过 `/gpus/clear_error` 端点手动清除

## 最佳实践

1. **共享 GPU 设置**：使用 `shared` 模式，75% 阈值以获得最佳吞吐量
2. **独占 GPU 任务**：对内存密集型工作负载使用 `exclusive` 模式
3. **监控**：定期检查 `/stats` 以发现队列积压
4. **维护**：硬件维护前将 GPU 设置为 `offline`
5. **错误恢复**：清除严重错误前先查看日志

## 相关文档

### 深入理解系统设计

如果你对以下问题有疑惑，请阅读设计文档：

- **GPU模式是什么时候切换的？** 🤔
- **为什么要区分exclusive和shared模式？** 🤔
- **任务类型(functional/performance)有什么作用？** 🤔
- **调度器如何决定任务分配？** 🤔
- **模式切换时正在运行的任务会怎样？** 🤔

推荐阅读：
- **[设计架构文档](DESIGN_ARCHITECTURE.md)** ⭐ - 完整的设计逻辑说明（English）
- **[核心概念说明](DESIGN_CORE_CONCEPTS_ZH.md)** ⭐ - GPU模式与任务类型详解（中文）

### 其他文档

- **[快速入门](QUICKSTART.md)** - 5分钟上手指南
- **[GPU配置指南](GPU_CONFIG_GUIDE.md)** - 配置GPU资源
- **[目录结构说明](DIRECTORY_STRUCTURE.md)** - 文件组织
- **[work_dir使用指南](WORK_DIR_GUIDE.md)** - 工作目录功能
- **[变更日志](CHANGELOG.md)** - 版本更新记录

### 示例代码

- **[examples/](../examples/)** - 各种使用场景的示例代码
  - `example_mode_switching.py` ⭐ - GPU模式切换示例（推荐）
  - `example_basic.py` - 基础用法
  - `example_concurrent_tasks.py` - 并发任务
  - 以及更多...

## 许可证

内部使用。
