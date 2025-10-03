# 最大并发任务数量控制

## 为什么需要？

在 shared（共享）模式下，仅依靠显存使用率来控制任务调度是不够的，因为：

1. **延迟显存占用**：很多任务在初始阶段显存占用很低，运行一段时间后才会大量占用显存
2. **累积效应**：如果只看当前显存，可能会调度过多任务，导致后续任务启动时显存不足
3. **任务预留**：需要为每个任务预留一定的资源空间

## 解决方案

在 v1.2.0 中，我们为 shared 模式的 GPU 添加了 `max_concurrent_tasks` 参数：

- **双重控制**：同时检查任务数量和显存使用率
- **可配置**：每个 GPU 可以独立设置最大并发任务数
- **合理预防**：在任务真正占用显存之前就进行控制

## 工作原理

### 任务接受逻辑

```python
def can_accept_task(self) -> bool:
    if self.status != GPUStatus.ONLINE:
        return False
    
    if self.mode == GPUMode.EXCLUSIVE:
        return len(self.running_tasks) == 0
    
    # Shared mode: 双重检查
    # 1. 检查任务数量
    if len(self.running_tasks) >= self.max_concurrent_tasks:
        return False  # 已达到最大任务数
    
    # 2. 检查显存阈值
    return self.current_memory_usage < self.memory_threshold
```

### 调度流程

```
任务提交
    ↓
检查 GPU 状态 (online?)
    ↓
检查任务数量 (<= max_concurrent_tasks?)
    ↓
检查显存使用率 (< memory_threshold?)
    ↓
接受任务 / 拒绝任务
```

## 配置方式

### 1. 配置文件 (gpu_resource.yml)

```yaml
gpus:
  - logical_id: 0
    name: "GPU 0"
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    max_concurrent_tasks: 3    # 最多3个并发任务（默认值）
    
  - logical_id: 1
    name: "GPU 1"
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.80
    max_concurrent_tasks: 6    # 最多6个并发任务（不同配置）
```

### 2. 通过 API 动态设置

```bash
# 设置 GPU 0 的最大并发任务数为 3
curl -X PUT http://localhost:8080/gpus/0/max_concurrent_tasks \
  -H "Content-Type: application/json" \
  -d '{"max_tasks": 3}'
```

### 3. 通过 Python 客户端

```python
from nvgpu.client import NVGPUClient

client = NVGPUClient()

# 设置最大并发任务数
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=3)

# 查看当前配置
gpu = client.get_gpu(0)
print(f"Max tasks: {gpu['max_concurrent_tasks']}")
print(f"Running: {gpu['running_task_count']}")
```

## 典型场景

### 场景 1：保守配置（防止 OOM）

```yaml
# 每个任务可能占用大量显存
max_concurrent_tasks: 2
memory_threshold: 0.70
```

**适用于**：
- 大模型训练/推理
- 高分辨率图像处理
- 显存密集型任务

### 场景 2：均衡配置（推荐，默认）

```yaml
# 平衡吞吐量和安全性
max_concurrent_tasks: 3
memory_threshold: 0.75
```

**适用于**：
- 一般深度学习任务
- 中等规模模型
- 混合工作负载

### 场景 3：高吞吐配置

```yaml
# 最大化GPU利用率
max_concurrent_tasks: 8
memory_threshold: 0.85
```

**适用于**：
- 轻量级任务
- 显存占用可预测的任务
- 快速迭代测试

## 实际示例

### 示例 1：观察任务排队

```python
# 设置较低的限制以观察效果
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=2)

# 提交5个任务
for i in range(5):
    task_id = client.submit_task(
        script_path="long_task.py",
        gpu_id=0
    )

# 观察任务状态
# 结果：2个运行，3个排队等待
```

### 示例 2：不同GPU不同配置

```python
# GPU 0: 显存密集型任务，限制2个
client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=2)

# GPU 1: 轻量级任务，允许6个
client.set_gpu_max_concurrent_tasks(gpu_id=1, max_tasks=6)
```

## API 端点

### 获取 GPU 信息

```bash
GET /gpus/{gpu_id}
```

响应：
```json
{
  "gpu_id": 0,
  "mode": "shared",
  "max_concurrent_tasks": 4,
  "running_task_count": 2,
  "memory_threshold": 0.75,
  "current_memory_usage": 0.45
}
```

### 设置最大并发任务数

```bash
PUT /gpus/{gpu_id}/max_concurrent_tasks
Content-Type: application/json

{
  "max_tasks": 3
}
```

## 监控和调优

### 查看当前负载

```bash
curl http://localhost:8080/gpus/0
```

关键指标：
- `running_task_count`: 当前运行任务数
- `max_concurrent_tasks`: 最大任务数限制
- `current_memory_usage`: 当前显存使用率

### 调优建议

1. **初始设置**：从保守值开始（如 4）
2. **监控观察**：
   - 如果经常出现 OOM → 减小 max_concurrent_tasks
   - 如果 GPU 利用率低 → 增大 max_concurrent_tasks
3. **压力测试**：提交批量任务，观察稳定性
4. **动态调整**：根据任务类型动态修改

### 监控脚本

```python
import time
from nvgpu.client import NVGPUClient

client = NVGPUClient()

while True:
    gpu = client.get_gpu(0)
    print(f"GPU 0: {gpu['running_task_count']}/{gpu['max_concurrent_tasks']} tasks, "
          f"{gpu['current_memory_usage']*100:.1f}% memory")
    time.sleep(5)
```

## 最佳实践

1. **分层配置**
   - 生产环境：保守配置（2-3个任务）
   - 测试环境：激进配置（6-8个任务）

2. **任务分类**
   - 显存密集型任务 → 专用 GPU，低 max_tasks
   - 计算密集型任务 → 共享 GPU，高 max_tasks

3. **动态调整**
   ```python
   # 白天高负载
   client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=2)
   
   # 夜间低负载
   client.set_gpu_max_concurrent_tasks(gpu_id=0, max_tasks=6)
   ```

4. **配合显存阈值**
   ```yaml
   # 保守组合
   max_concurrent_tasks: 2
   memory_threshold: 0.70
   
   # 激进组合
   max_concurrent_tasks: 8
   memory_threshold: 0.90
   ```

## 与 exclusive 模式对比

| 特性 | exclusive 模式 | shared + max_concurrent_tasks |
|------|---------------|------------------------------|
| 任务数量 | 固定1个 | 可配置（1-N个） |
| 显存利用率 | 可能较低 | 更高 |
| 任务隔离 | 完全隔离 | 部分隔离 |
| 灵活性 | 低 | 高 |
| 适用场景 | 关键任务 | 一般任务 |

## 常见问题

**Q: max_concurrent_tasks 设置多少合适？**  
A: 默认值是 3，这是比较稳妥的配置。根据实际任务的显存占用可以调整：显存密集型任务可设为 2，轻量级任务可设为 6-8。

**Q: 达到 max_concurrent_tasks 后，新任务怎么处理？**  
A: 新任务会被放入该 GPU 的队列中等待，当运行任务数降低后自动执行。

**Q: exclusive 模式下 max_concurrent_tasks 有效吗？**  
A: 无效。exclusive 模式固定为1个任务，不受 max_concurrent_tasks 限制。

**Q: 可以为不同 GPU 设置不同的限制吗？**  
A: 可以！这正是设计目标，每个 GPU 可以独立配置。

**Q: 运行中可以修改 max_concurrent_tasks 吗？**  
A: 可以通过 API 动态修改，立即生效。但已运行的任务不会被中断。

## 示例代码

完整示例见：
- `examples/example_concurrent_tasks.py` - 并发任务控制演示
- `examples/example_gpu_management.py` - GPU 配置管理

运行示例：
```bash
python examples/example_concurrent_tasks.py
```

## 总结

`max_concurrent_tasks` 为 shared 模式提供了更精细的任务调度控制：

✅ **防止过载**：在显存占用之前就进行控制  
✅ **灵活配置**：每个 GPU 独立设置  
✅ **动态调整**：运行时随时修改  
✅ **双重保护**：结合显存阈值，双重保障  

这是对原有显存阈值控制的重要补充，特别适合处理延迟显存占用的任务！

