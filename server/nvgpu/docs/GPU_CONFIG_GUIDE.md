# GPU 配置指南

## 概述

NVGPU 服务器支持通过 YAML 配置文件管理 GPU 资源，包括处理 nvidia-smi ID 和 CUDA_VISIBLE_DEVICES 的映射。

## 配置文件：gpu_resource.yml

### 基本结构

```yaml
gpus:
  - logical_id: 0              # 系统内部逻辑 ID
    nvidia_smi_id: 0           # nvidia-smi 显示的 GPU 索引
    cuda_visible_id: 0         # CUDA_VISIBLE_DEVICES 使用的 ID
    name: "GPU Name"           # GPU 名称
    uuid: "GPU-xxx"            # GPU UUID（可选）
    memory_gb: 48              # 显存大小（GB）
    enabled: true              # 是否启用
    default_mode: "shared"     # 默认模式
    memory_threshold: 0.75     # 显存阈值

server:
  auto_register_gpus: true     # 启动时自动注册
  use_gpu_uuid: true           # 使用 UUID 识别（更可靠）
```

## GPU ID 映射说明

### 三种 GPU ID

1. **logical_id** - 系统内部逻辑 ID
   - 用于任务调度、API 调用
   - 通常从 0 开始递增
   - 示例：`0, 1, 2, 3`

2. **nvidia_smi_id** - nvidia-smi 显示的索引
   - 运行 `nvidia-smi` 命令看到的 GPU 索引
   - 用于获取 GPU 状态和显存信息
   - 可能与逻辑 ID 不同

3. **cuda_visible_id** - CUDA_VISIBLE_DEVICES 值
   - 设置给任务的 CUDA_VISIBLE_DEVICES 环境变量
   - CUDA 程序实际看到的 GPU ID
   - 在某些配置下可能与前两者不同

### 为什么需要映射？

在某些情况下，这三个 ID 可能不一致：

1. **多 GPU 系统** - 不同的枚举顺序
2. **MIG 分区** - NVIDIA Multi-Instance GPU
3. **虚拟化环境** - GPU pass-through
4. **自定义配置** - 用户特定的 GPU 顺序

## 当前系统配置

您的系统有两张 **RTX 6000 Ada Generation** GPU：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "NVIDIA RTX 6000 Ada Generation"
    uuid: "GPU-1b8c43a8-35ba-a01b-a2ee-0d6fe8f1f60f"
    memory_gb: 48
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
    
  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    name: "NVIDIA RTX 6000 Ada Generation"
    uuid: "GPU-78b50b8e-ef33-b3fd-2c52-e3f135990f31"
    memory_gb: 48
    enabled: true
    default_mode: "shared"
    memory_threshold: 0.75
```

## 使用方式

### 1. 使用配置文件启动（推荐）

```bash
# 使用默认配置文件 gpu_resource.yml
python main.py

# 指定配置文件
python main.py --gpu-config /path/to/custom_gpu_config.yml
```

配置文件中 `enabled: true` 的 GPU 会自动注册。

### 2. 命令行指定 GPU（覆盖配置文件）

```bash
# 只使用 GPU 0 和 1
python main.py --gpus 0 1

# 指定模式和阈值
python main.py --gpus 0 1 --gpu-mode exclusive --memory-threshold 0.8
```

### 3. 混合使用

```bash
# 使用配置文件，但命令行覆盖
python main.py --gpu-config my_gpus.yml --gpus 0
```

## 配置示例

### 示例 1：标准配置（ID 一致）

最常见的情况，所有 ID 相同：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "GPU 0"
    enabled: true
    
  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    name: "GPU 1"
    enabled: true
```

### 示例 2：反向映射

nvidia-smi 和 CUDA 顺序相反：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 1    # CUDA 看到的是 1
    name: "GPU 0"
    enabled: true
    
  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 0    # CUDA 看到的是 0
    name: "GPU 1"
    enabled: true
```

### 示例 3：部分 GPU

只使用部分 GPU：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    enabled: true         # 使用
    
  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    enabled: false        # 不使用
    
  - logical_id: 2
    nvidia_smi_id: 2
    cuda_visible_id: 2
    enabled: true         # 使用
```

### 示例 4：不同配置

为不同 GPU 设置不同参数：

```yaml
gpus:
  - logical_id: 0
    nvidia_smi_id: 0
    cuda_visible_id: 0
    name: "High-end GPU"
    enabled: true
    default_mode: "shared"     # 共享模式
    memory_threshold: 0.75     # 75% 阈值
    
  - logical_id: 1
    nvidia_smi_id: 1
    cuda_visible_id: 1
    name: "Dedicated GPU"
    enabled: true
    default_mode: "exclusive"  # 独占模式
    memory_threshold: 0.9      # 90% 阈值
```

## 验证配置

### 1. 检查 nvidia-smi ID

```bash
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

### 2. 验证映射

创建测试脚本 `test_gpu_mapping.py`：

```python
import os
import torch

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
```

提交测试任务：

```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "test_gpu_mapping.py",
    "gpu_id": 0
  }'
```

## 严重错误暂停机制

### 新的自动恢复机制

- **暂停时长**: 60 秒（1 分钟）
- **触发条件**: 检测到 GPU 严重错误
- **暂停行为**: 停止所有新任务调度，已运行任务继续
- **自动恢复**: 60 秒后自动恢复调度

### 配置暂停时长

修改 `config.py`：

```python
class ServerConfig:
    error_pause_duration: int = 60  # 暂停时长（秒）
```

### 手动控制

```bash
# 手动触发严重错误（测试用）
curl -X POST http://localhost:8080/gpus/0/error \
  -H "Content-Type: application/json" \
  -d '{"error_message": "Test error"}'

# 手动清除错误（不等待自动恢复）
curl -X POST http://localhost:8080/gpus/clear_error
```

## 故障排查

### 问题 1: GPU 未自动注册

**检查**:
- 配置文件路径正确
- `enabled: true`
- `auto_register_gpus: true`

**解决**:
```bash
# 查看日志
tail -f logs/nvgpu_server.log

# 手动指定 GPU
python main.py --gpus 0 1
```

### 问题 2: GPU ID 映射错误

**症状**: 任务在错误的 GPU 上运行

**检查**:
```bash
# 在任务中打印环境变量
echo $CUDA_VISIBLE_DEVICES
```

**解决**: 修改配置文件中的 `cuda_visible_id`

### 问题 3: 显存监控不准确

**原因**: `nvidia_smi_id` 配置错误

**解决**: 
1. 运行 `nvidia-smi` 查看正确的索引
2. 更新配置文件中的 `nvidia_smi_id`

## 最佳实践

1. **使用 UUID** - 比索引更可靠
   ```yaml
   uuid: "GPU-xxx"
   ```

2. **记录 GPU 信息** - 便于维护
   ```yaml
   name: "Node1-GPU0-RTX6000"
   memory_gb: 48
   ```

3. **版本控制** - 将配置文件加入 git
   ```bash
   git add gpu_resource.yml
   ```

4. **测试映射** - 启动后验证 GPU 映射正确

5. **环境特定配置** - 不同环境使用不同配置文件
   ```bash
   python main.py --gpu-config prod_gpus.yml
   python main.py --gpu-config dev_gpus.yml
   ```

## 配置模板

### 生成配置模板

```bash
# 自动生成当前系统的 GPU 配置
python -c "
import pynvml
import yaml

pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()

gpus = []
for i in range(count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    gpus.append({
        'logical_id': i,
        'nvidia_smi_id': i,
        'cuda_visible_id': i,
        'name': name,
        'uuid': uuid,
        'memory_gb': int(mem.total / 1e9),
        'enabled': True,
        'default_mode': 'shared',
        'memory_threshold': 0.75
    })

config = {
    'gpus': gpus,
    'server': {
        'auto_register_gpus': True,
        'use_gpu_uuid': True
    }
}

print(yaml.dump(config, default_flow_style=False))
"
```

## 参考

- [NVIDIA Management Library (NVML)](https://developer.nvidia.com/nvidia-management-library-nvml)
- [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)

