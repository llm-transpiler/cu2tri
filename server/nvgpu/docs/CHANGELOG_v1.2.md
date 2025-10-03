# Version 1.2.0 - 最大并发任务数控制

## 🎯 核心改进

### 新增：最大并发任务数控制

在 shared 模式下，仅依靠显存使用率控制是不够的！

**问题**：
- 任务初始阶段显存占用可能很低
- 过一段时间后才会占用大量显存
- 如果只看当前显存，可能会调度过多任务

**解决方案**：
- ✅ 添加 `max_concurrent_tasks` 参数
- ✅ 双重检查：任务数量 + 显存阈值
- ✅ 每个 GPU 可独立配置
- ✅ 支持动态调整

## 📦 新增内容

### 1. GPU 模型更新

```python
@dataclass
class GPU:
    gpu_id: int
    mode: GPUMode = GPUMode.SHARED
    memory_threshold: float = 0.75
    max_concurrent_tasks: int = 3  # 新增
    running_tasks: list[str]
    # ...
```

### 2. 配置文件支持

```yaml
# gpu_resource.yml
gpus:
  - logical_id: 0
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3    # 新增配置项
```

### 3. 新增 API 端点

```
PUT /gpus/{gpu_id}/max_concurrent_tasks
```

请求：
```json
{
  "max_tasks": 6
}
```

### 4. GPU 信息响应更新

```json
{
  "gpu_id": 0,
  "mode": "shared",
  "max_concurrent_tasks": 3,
  "running_task_count": 2,     # 新增
  "running_tasks": [...],
  "memory_threshold": 0.75,
  "current_memory_usage": 0.45
}
```

### 5. 客户端库更新

```python
client = NVGPUClient()

# 新增方法
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=6)

# GPU 信息包含新字段
gpu = client.get_gpu(0)
print(gpu['max_concurrent_tasks'])
print(gpu['running_task_count'])
```

### 6. 新增文档

- **MAX_CONCURRENT_TASKS.md** - 详细说明和最佳实践
- **examples/example_concurrent_tasks.py** - 完整示例

## 🔄 工作原理

### 任务接受逻辑

```python
def can_accept_task(self) -> bool:
    if self.status != GPUStatus.ONLINE:
        return False
    
    if self.mode == GPUMode.EXCLUSIVE:
        return len(self.running_tasks) == 0
    
    # Shared 模式：双重检查
    # 1. 检查任务数量（防止过载）
    if len(self.running_tasks) >= self.max_concurrent_tasks:
        return False
    
    # 2. 检查显存阈值
    return self.current_memory_usage < self.memory_threshold
```

### 调度保护

```
任务提交
    ↓
检查在线状态
    ↓
检查任务数量 (<= max_concurrent_tasks?) ← 新增检查
    ↓
检查显存使用率 (< memory_threshold?)
    ↓
接受/拒绝任务
```

## 📋 更新的文件

1. **models.py** - 添加 `max_concurrent_tasks` 字段
2. **config.py** - 添加 `default_max_concurrent_tasks` 配置
3. **gpu_manager.py** - 添加设置方法
4. **api_server.py** - 添加 API 端点
5. **client.py** - 添加客户端方法
6. **gpu_resource.yml** - 更新配置文件
7. **gpu_config_loader.py** - 支持加载新配置
8. **main.py** - 传递新参数
9. **README.md** - 更新文档
10. **QUICKSTART.md** - 更新快速开始

## 📚 新增文件

1. **MAX_CONCURRENT_TASKS.md** (6.5K) - 详细文档
2. **examples/example_concurrent_tasks.py** - 示例代码
3. **CHANGELOG_v1.2.md** - 本文件

## 🚀 使用示例

### 配置文件方式

```yaml
gpus:
  - logical_id: 0
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3
    
  - logical_id: 1
    default_mode: "shared"
    memory_threshold: 0.80
    max_concurrent_tasks: 6  # 不同GPU不同配置
```

### API 方式

```bash
# 设置 GPU 0 最多3个并发任务
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 3}'
```

### Python 客户端方式

```python
from nvgpu.client import NVGPUClient

client = NVGPUClient()

# 设置最大并发任务数
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=3)

# 查看配置
gpu = client.get_gpu(0)
print(f"Running: {gpu['running_task_count']}/{gpu['max_concurrent_tasks']}")
```

## 🎓 使用场景

### 场景 1：显存密集型任务

```yaml
max_concurrent_tasks: 2
memory_threshold: 0.70
```

**适用**：大模型训练、高分辨率图像处理

### 场景 2：均衡配置（推荐）

```yaml
max_concurrent_tasks: 4
memory_threshold: 0.75
```

**适用**：一般深度学习任务

### 场景 3：高吞吐量

```yaml
max_concurrent_tasks: 8
memory_threshold: 0.85
```

**适用**：轻量级任务、快速测试

## ✅ 向后兼容

完全向后兼容！

- 默认值：`max_concurrent_tasks = 3`
- 如果不设置，使用默认值
- 旧的配置文件仍然有效
- API 保持兼容

## 📊 效果对比

### 之前（仅显存控制）

```
GPU 显存: 10% → 调度3个任务
5秒后...
GPU 显存: 85% → 超出阈值，但任务已启动
10秒后...
GPU 显存: 95% → OOM 错误！
```

### 现在（双重控制）

```
GPU 任务数: 2 → 检查通过
GPU 显存: 10% → 检查通过 → 调度任务
任务数达到 max_concurrent_tasks=2 → 停止调度
等待任务完成后才接受新任务
```

## 🔧 迁移指南

### 从 v1.1.0 升级

1. **更新代码**（无需修改）
   - 服务器自动使用默认值（3）

2. **更新配置文件**（可选）
   ```yaml
   # 添加 max_concurrent_tasks 字段
   max_concurrent_tasks: 3
   ```

3. **测试调整**
   - 监控 `running_task_count`
   - 根据实际情况调整参数

### 推荐配置

```yaml
# 保守配置（显存密集）
max_concurrent_tasks: 2
memory_threshold: 0.70

# 标准配置（推荐，默认值）
max_concurrent_tasks: 3
memory_threshold: 0.75

# 激进配置（轻量任务）
max_concurrent_tasks: 8
memory_threshold: 0.85
```

## 📖 相关文档

- [MAX_CONCURRENT_TASKS.md](MAX_CONCURRENT_TASKS.md) - 详细说明
- [README.md](README.md) - 完整文档
- [GPU_CONFIG_GUIDE.md](GPU_CONFIG_GUIDE.md) - GPU 配置
- [QUICKSTART.md](QUICKSTART.md) - 快速开始

## 🎯 总结

v1.2.0 为 shared 模式添加了更精细的任务调度控制：

✅ **双重保护**：任务数量 + 显存阈值  
✅ **提前预防**：在显存占用前就控制  
✅ **灵活配置**：每个 GPU 独立设置  
✅ **动态调整**：运行时随时修改  
✅ **完全兼容**：无需修改现有代码  

这是对显存阈值控制的重要补充，特别适合处理延迟显存占用的任务！

