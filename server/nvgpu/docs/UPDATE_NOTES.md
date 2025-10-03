# 更新说明 (v1.1.0)

## 主要更新

### 1. ✅ 严重错误暂停机制优化

**之前**：
- 暂停时长：5 分钟
- 需要手动恢复

**现在**：
- 暂停时长：**1 分钟**（60秒）
- **自动恢复**调度
- 仍支持手动立即恢复

**配置**：
```python
# config.py
error_pause_duration: int = 60  # 可修改
```

### 2. ✅ GPU 配置文件支持

新增 `gpu_resource.yml` 配置文件，支持：

- 定义可用 GPU 及其属性
- GPU ID 映射（nvidia-smi ↔ CUDA_VISIBLE_DEVICES）
- 每个 GPU 独立配置（模式、阈值）
- 启动时自动注册

**当前系统配置**：

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

### 3. ✅ GPU ID 映射系统

**为什么需要？**

在某些系统中，nvidia-smi 显示的 GPU 索引和 CUDA 程序看到的索引可能不一致。

**支持的映射**：

- `logical_id`: 系统内部使用的逻辑 ID
- `nvidia_smi_id`: nvidia-smi 命令中的索引
- `cuda_visible_id`: CUDA_VISIBLE_DEVICES 使用的值

**自动处理**：

- 任务执行时自动设置正确的 CUDA_VISIBLE_DEVICES
- GPU 监控时使用正确的 nvidia-smi 索引
- 对用户透明

## 使用方式

### 启动服务器

```bash
# 使用配置文件（推荐）
python main.py

# 指定配置文件
python main.py --gpu-config my_gpus.yml

# 命令行指定（覆盖配置文件）
python main.py --gpus 0 1
```

### 验证 GPU 映射

```bash
# 提交测试任务
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "test_gpu_mapping.py",
    "gpu_id": 0
  }'

# 查看日志确认 CUDA_VISIBLE_DEVICES 正确
```

## 新增文件

1. **gpu_resource.yml** - GPU 配置文件
2. **gpu_config_loader.py** - 配置加载器
3. **GPU_CONFIG_GUIDE.md** - 配置详细指南
4. **CHANGELOG.md** - 变更日志
5. **test_gpu_mapping.py** - GPU 映射测试脚本

## 更新文件

1. **config.py** - 错误暂停时长改为 60 秒
2. **gpu_manager.py** - 支持自动恢复，GPU ID 映射
3. **task_runner.py** - 使用映射的 CUDA ID
4. **main.py** - 支持配置文件加载
5. **requirements.txt** - 新增 pyyaml 依赖
6. **README.md** - 更新文档
7. **QUICKSTART.md** - 更新快速开始

## 新增依赖

```bash
pip install pyyaml>=6.0
```

## 向后兼容

✅ 完全向后兼容！

- 所有旧的命令行参数仍然工作
- 如果没有配置文件，按原方式工作
- API 接口未改变
- 任务提交格式未改变

## 推荐做法

1. **使用配置文件** - 更清晰、可维护
2. **启用自动注册** - 减少启动参数
3. **配置文件版本控制** - 便于团队协作
4. **使用 GPU UUID** - 更可靠的识别

## 文档

- 📖 [README.md](README.md) - 完整文档
- 🚀 [QUICKSTART.md](QUICKSTART.md) - 快速开始
- 🔧 [GPU_CONFIG_GUIDE.md](GPU_CONFIG_GUIDE.md) - GPU 配置指南
- 📋 [CHANGELOG.md](CHANGELOG.md) - 完整变更日志

## 示例

### 示例 1：简单启动（使用配置文件）

```bash
# 1. 检查配置文件
cat gpu_resource.yml

# 2. 直接启动（自动注册两个 GPU）
python main.py

# 3. 验证
curl http://localhost:8080/gpus
```

### 示例 2：自定义配置

```bash
# 1. 复制模板
cp gpu_resource.yml my_config.yml

# 2. 编辑配置
vim my_config.yml

# 3. 使用自定义配置
python main.py --gpu-config my_config.yml
```

### 示例 3：测试 GPU 映射

```bash
# 1. 启动服务器
python main.py

# 2. 提交测试任务
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "functional",
    "script_path": "test_gpu_mapping.py",
    "work_dir": "/workspace/server/nvgpu"
  }'

# 3. 查看结果
curl http://localhost:8080/tasks/{task_id}
```

### 示例 4：测试错误恢复

```bash
# 1. 触发错误
curl -X POST http://localhost:8080/gpus/0/error \
  -H "Content-Type: application/json" \
  -d '{"error_message": "Test error"}'

# 2. 查看状态
curl http://localhost:8080/stats

# 3. 等待 60 秒，自动恢复
# 或手动立即恢复：
curl -X POST http://localhost:8080/gpus/clear_error
```

## 问题排查

### 配置文件未加载

**检查**：
```bash
# 1. 文件是否存在
ls -l gpu_resource.yml

# 2. YAML 格式是否正确
python -c "import yaml; yaml.safe_load(open('gpu_resource.yml'))"

# 3. 查看日志
tail -f logs/nvgpu_server.log
```

### GPU ID 映射错误

**检查**：
```bash
# 1. 查看 nvidia-smi 索引
nvidia-smi --query-gpu=index,name,uuid --format=csv

# 2. 提交测试任务查看 CUDA_VISIBLE_DEVICES
python examples/example_basic.py

# 3. 检查任务日志
cat logs/tasks/{task_id}.log | grep CUDA_VISIBLE_DEVICES
```

### 依赖问题

```bash
# 重新安装依赖
pip install -r requirements.txt --upgrade
```

## 反馈

如有问题，请查看：
- 服务器日志：`logs/nvgpu_server.log`
- 任务日志：`logs/tasks/{task_id}.log`
- 文档：`GPU_CONFIG_GUIDE.md`

