# NVGPU Server 快速开始指南

## 项目结构

```
/workspace/server/nvgpu/
├── config.py              # 服务器配置
├── models.py              # 数据模型（GPU、Task）
├── logger.py              # 统一日志配置
├── gpu_manager.py         # GPU 管理器（监控、状态管理）
├── task_queue.py          # 任务队列（全局队列 + 每个GPU队列）
├── task_runner.py         # 任务执行器（运行Python脚本）
├── scheduler.py           # 任务调度器（FIFO调度）
├── api_server.py          # REST API 服务器
├── main.py                # 主入口
├── requirements.txt       # Python依赖
├── example_test_script.py # 示例测试脚本
├── test_api.sh            # API测试脚本
└── README.md              # 完整文档
```

## 核心设计

### 1. GPU 模式
- **exclusive**: 每次只能运行一个任务（独占模式）
- **shared**: 可运行多个任务，双重控制：
  - 最大并发任务数（默认：3个）
  - 显存使用阈值（默认：75%）

### 2. 任务类型
- **functional**: 功能验证和测试
- **performance**: 性能测试和基准测试

### 3. 架构流程
```
提交任务 → 全局队列 → 调度器分配GPU → GPU队列 → 任务执行 → 更新状态
                              ↓
                        GPU管理器监控
                   （显存、状态、错误检测）
```

### 4. 严重错误处理
- 当检测到GPU严重错误时，自动暂停所有任务调度
- **自动恢复**：暂停 60 秒后自动恢复调度
- 也可手动调用 `/gpus/clear_error` 立即恢复

## 安装依赖

```bash
cd /workspace/server/nvgpu
pip install -r requirements.txt
```

依赖：
- `flask`: REST API服务器
- `pynvml`: NVIDIA GPU监控
- `pyyaml`: YAML配置文件支持

## GPU 配置文件

**重要**：现在支持通过 `gpu_resource.yml` 配置 GPU！

当前系统已配置两张 RTX 6000 Ada Generation GPU：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "NVIDIA RTX 6000 Ada Generation"
    memory_gb: 48
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3  # 最多3个并发任务
  
  - logical_id: 1
    # ... (GPU 1 配置)
```

详见：
- [GPU_CONFIG_GUIDE.md](GPU_CONFIG_GUIDE.md) - GPU 配置指南
- [MAX_CONCURRENT_TASKS.md](MAX_CONCURRENT_TASKS.md) - 并发任务控制详解

## 启动服务器

### 方式 1：使用配置文件（推荐）
```bash
# 使用默认配置文件 gpu_resource.yml
python main.py
```

配置文件中 `enabled: true` 的 GPU 会自动注册。

### 方式 2：命令行指定
```bash
python main.py --gpus 0 1 2 3
```

### 方式 3：自定义配置文件
```bash
python main.py --gpu-config /path/to/custom_gpu_config.yml
```

### 方式 4：命令行自定义
```bash
python main.py \
  --gpus 0 1 \
  --gpu-mode shared \
  --memory-threshold 0.8 \
  --port 8080 \
  --log-level DEBUG
```

### 参数说明
- `--gpu-config`: GPU配置文件路径 (默认: gpu_resource.yml)
- `--gpus`: 启动时注册的GPU ID列表 (覆盖配置文件)
- `--gpu-mode`: 默认GPU模式 (exclusive/shared)
- `--memory-threshold`: 默认显存阈值 (0-1)
- `--port`: API服务器端口
- `--host`: 服务器地址
- `--log-level`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `--log-file`: 日志文件路径

## API 使用示例

### 1. 提交任务

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "/path/to/your_test.py",
    "work_dir": "/path/to/work/directory",
    "args": ["--arg1", "value1"],
    "gpu_id": 0
  }'
```

响应：
```json
{
  "success": true,
  "task_id": "uuid-string",
  "task": { ... }
}
```

### 2. 查询任务状态

```bash
curl http://localhost:8080/tasks/{task_id}
```

### 3. GPU 管理

#### 查看所有 GPU
```bash
curl http://localhost:8080/gpus
```

#### 修改GPU模式
```bash
curl -X PUT http://localhost:8080/gpus/0/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "exclusive"}'
```

#### GPU 上线/下线
```bash
# 下线（维护）
curl -X PUT http://localhost:8080/gpus/0/status \
  -H "Content-Type: application/json" \
  -d '{"status": "offline"}'

# 上线
curl -X PUT http://localhost:8080/gpus/0/status \
  -H "Content-Type: application/json" \
  -d '{"status": "online"}'
```

#### 修改显存阈值
```bash
curl -X PUT http://localhost:8080/gpus/0/memory_threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.8}'
```

#### 修改最大并发任务数
```bash
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 6}'
```

### 4. 监控统计

```bash
curl http://localhost:8080/stats
```

响应示例：
```json
{
  "queue": {
    "global_queue_size": 5,
    "total_tasks": 100,
    "pending": 5,
    "queued": 10,
    "running": 8,
    "completed": 75,
    "failed": 2,
    "gpu_queues": {
      "0": 3,
      "1": 2
    }
  },
  "gpus": {
    "total_gpus": 4,
    "online_gpus": 3,
    "severe_error_active": false
  }
}
```

## 任务脚本格式

你的测试脚本应该遵循以下格式：

```python
#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-perf", action="store_true")
    # 其他参数...
    args = parser.parse_args()
    
    # 你的测试代码
    # ...
    
    # 返回0表示成功
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**重要特性**：
- 脚本会自动设置 `CUDA_VISIBLE_DEVICES` 环境变量
- 所有参数通过 args 传递，不会自动添加额外参数
- stdout 和 stderr 会被捕获并保存
- 超时时间：默认 600 秒（10分钟）

## 测试脚本

### 运行示例测试
```bash
# 确保服务器正在运行，然后执行
./test_api.sh
```

### 手动提交示例任务
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
    "work_dir": "/workspace/server/nvgpu",
    "args": []
  }'
```

## 日志文件

- 服务器日志：`logs/nvgpu_server.log`
- 任务日志：`logs/tasks/{task_id}.log`

每个任务日志包含：
- 任务信息（ID、类型、GPU等）
- 执行命令
- 时间戳
- 完整的 stdout 和 stderr

## 错误处理

### GPU 错误检测
系统会自动检测任务输出中的 GPU 错误关键词：
- "cuda error"
- "gpu error"
- "out of memory"

### 手动触发严重错误（测试用）
```bash
curl -X POST http://localhost:8080/gpus/0/error \
  -H "Content-Type: application/json" \
  -d '{"error_message": "Manual test error"}'
```

### 清除错误状态（立即恢复）
```bash
# 严重错误会在60秒后自动恢复
# 也可以手动立即清除
curl -X POST http://localhost:8080/gpus/clear_error
```

## 最佳实践

1. **共享GPU设置**：使用 `shared` 模式，设置合适的显存阈值（推荐75%-80%）
2. **独占GPU任务**：对于显存密集型任务，使用 `exclusive` 模式
3. **定期监控**：通过 `/stats` 端点监控队列和GPU状态
4. **维护操作**：维护前将GPU状态设为 `offline`
5. **错误恢复**：出现错误后检查日志再清除错误状态

## 常见问题

**Q: 如何添加新的GPU？**
```bash
curl -X POST http://localhost:8080/gpus/register \
  -H "Content-Type: application/json" \
  -d '{"gpu_id": 4, "mode": "shared", "memory_threshold": 0.75}'
```

**Q: 如何取消排队中的任务？**
```bash
curl -X POST http://localhost:8080/tasks/{task_id}/cancel
```

**Q: pynvml 不可用怎么办？**  
A: 服务器会禁用GPU监控但仍可正常运行，只是无法自动检测显存使用情况。

**Q: 如何查看任务日志？**  
A: 日志文件路径在任务详情的 `log_file` 字段中，直接读取该文件。

**Q: 如何配置 GPU ID 映射？**  
A: 编辑 `gpu_resource.yml`，设置每个 GPU 的 `nvidia_smi_id` 和 `cuda_visible_id`。详见 [GPU_CONFIG_GUIDE.md](GPU_CONFIG_GUIDE.md)

**Q: 严重错误多久会自动恢复？**  
A: 默认 60 秒后自动恢复。可在 `config.py` 中修改 `error_pause_duration` 参数。

## 技术特点

- ✅ 简洁的架构设计
- ✅ 线程安全的队列和GPU管理
- ✅ 统一的日志系统
- ✅ RESTful API 设计
- ✅ 完整的错误处理
- ✅ 实时GPU监控
- ✅ 灵活的配置选项
- ✅ 经典FIFO调度算法

---

更多详细信息请参考 [README.md](README.md)

