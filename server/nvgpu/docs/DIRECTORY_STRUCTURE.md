# NVGPU 目录结构说明

本文档说明 NVGPU 服务器的目录结构和路径处理策略。

## 目录结构

```
nvgpu/
├── README.md                   # 项目简介和快速导航
├── requirements.txt            # Python依赖包列表
├── __init__.py                 # Python包初始化文件
│
├── main.py                     # 服务器主入口 ⭐
├── config.py                   # 全局配置管理
├── models.py                   # 数据模型定义
├── logger.py                   # 统一日志系统
│
├── api_server.py               # REST API服务器
├── client.py                   # Python客户端库
│
├── gpu_manager.py              # GPU资源管理器
├── gpu_config_loader.py        # GPU配置加载器
├── task_queue.py               # 任务队列管理
├── task_runner.py              # 任务执行引擎
├── scheduler.py                # 任务调度器
│
├── configs/                    # 配置文件目录 📁
│   └── gpu_resource.yml        # GPU资源配置文件
│
├── docs/                       # 文档目录 📁
│   ├── README.md               # 完整功能文档
│   ├── QUICKSTART.md           # 快速入门指南
│   ├── GPU_CONFIG_GUIDE.md     # GPU配置详细说明
│   ├── LOG_HANDLING.md         # 日志处理机制
│   ├── TASK_CANCELLATION.md    # 任务取消功能
│   ├── MAX_CONCURRENT_TASKS.md # 并发任务限制说明
│   ├── DIRECTORY_STRUCTURE.md  # 本文档
│   ├── CHANGELOG.md            # 完整变更日志
│   ├── CHANGELOG_v1.2.md       # v1.2 版本变更
│   ├── UPDATE_NOTES.md         # 更新说明
│   ├── UPDATES_v1.3.md         # v1.3 版本更新
│   └── PROJECT_SUMMARY.md      # 项目总结
│
├── scripts/                    # 工具脚本目录 📁
│   ├── test_api.sh             # API接口测试脚本
│   ├── test_gpu_mapping.py     # GPU ID映射测试工具
│   └── example_test_script.py  # 示例测试脚本
│
├── examples/                   # 使用示例目录 📁
│   ├── README.md               # 示例说明
│   ├── example_basic.py        # 基础使用示例
│   ├── example_batch_submit.py # 批量提交示例
│   ├── example_concurrent_tasks.py  # 并发任务示例
│   ├── example_custom_env.py   # 自定义环境变量
│   ├── example_error_handling.py    # 错误处理示例
│   ├── example_gpu_management.py    # GPU管理示例
│   ├── example_log_handling.py      # 日志处理示例
│   └── example_cancel_running_task.py  # 任务取消示例
│
├── test_scripts/               # 测试脚本目录 📁
│   ├── simple_functional_test.py    # 简单功能测试
│   ├── failing_test.py              # 失败场景测试
│   ├── long_running_task.py         # 长时间运行任务
│   ├── gpu_memory_intensive.py      # GPU内存密集测试
│   ├── memory_stress_test.py        # 内存压力测试
│   ├── multi_gpu_test.py            # 多GPU测试
│   └── performance_benchmark.py     # 性能基准测试
│
└── logs/                       # 日志目录（运行时生成） 📁
    ├── nvgpu_server.log        # 历史日志（追加模式）
    ├── nvgpu_server_YYYYMMDD_HHMMSS.log  # 会话日志（时间戳）
    └── tasks/                  # 任务日志子目录
        ├── {task_id}.log       # 任务摘要日志
        ├── {task_id}.stdout    # 任务标准输出
        └── {task_id}.stderr    # 任务标准错误输出
```

## 路径处理策略

### 绝对路径基准

NVGPU 服务器使用 **绝对路径** 作为基准，以确保无论从哪个目录启动服务器，所有路径都能正确解析。

基准路径定义：
```python
# main.py
NVGPU_ROOT = Path(__file__).parent.resolve()
```

这意味着 `NVGPU_ROOT` 永远指向 `main.py` 所在的目录（即 `nvgpu/` 文件夹）。

### 各类路径说明

#### 1. 配置文件路径

**默认路径：** `{NVGPU_ROOT}/configs/gpu_resource.yml`

可以通过命令行参数指定：
```bash
python main.py --gpu-config /path/to/custom_config.yml
```

路径解析规则：
- 如果是绝对路径，直接使用
- 如果是相对路径，优先相对于 `NVGPU_ROOT`，其次相对于当前工作目录

#### 2. 日志文件路径

**会话日志：** `{NVGPU_ROOT}/logs/nvgpu_server_YYYYMMDD_HHMMSS.log`
- 每次启动服务器创建新文件
- 包含当前会话的所有日志

**历史日志：** `{NVGPU_ROOT}/logs/nvgpu_server.log`
- 追加模式，跨会话累积
- 包含所有历史日志

**任务日志：** `{NVGPU_ROOT}/logs/tasks/{task_id}.*`
- 每个任务生成三个文件：`.log`（摘要）、`.stdout`、`.stderr`

可以通过命令行参数自定义：
```bash
python main.py --log-file /custom/path/server.log
```

#### 3. 任务脚本路径

提交任务时，脚本路径会被转换为绝对路径：
```python
script_abs_path = os.path.abspath(task.script_path)
```

这确保任务执行器能够找到正确的脚本文件。

## 从不同目录运行

由于使用了绝对路径，服务器可以从任意目录启动：

```bash
# 从 nvgpu 目录启动
cd /workspace/server/nvgpu
python main.py

# 从其他目录启动（推荐使用绝对路径）
cd /tmp
python /workspace/server/nvgpu/main.py

# 使用相对路径（需确保路径正确）
cd /workspace/server
python nvgpu/main.py
```

所有日志和配置文件都会正确写入到 `{NVGPU_ROOT}` 相对的位置。

## 配置文件查找顺序

当指定配置文件时（如 `--gpu-config configs/gpu_resource.yml`），查找顺序为：

1. 检查是否为绝对路径
   - 是：直接使用
   - 否：继续下一步

2. 相对于 `NVGPU_ROOT` 查找
   - 检查 `{NVGPU_ROOT}/configs/gpu_resource.yml`
   - 存在：使用此路径

3. 相对于当前工作目录查找
   - 检查 `{CWD}/configs/gpu_resource.yml`
   - 存在：使用此路径

4. 配置文件未找到
   - 服务器将不加载 GPU 配置
   - 需要通过 API 或命令行参数手动注册 GPU

## 最佳实践

1. **使用默认路径**
   - 配置文件放在 `configs/` 目录
   - 测试脚本放在 `test_scripts/` 目录
   - 工具脚本放在 `scripts/` 目录

2. **启动服务器**
   - 推荐直接在 `nvgpu/` 目录下启动：`python main.py`
   - 或使用绝对路径：`python /full/path/to/main.py`

3. **提交任务**
   - 使用绝对路径指定脚本：`/workspace/test.py`
   - 或确保工作目录正确，使用相对路径

4. **查看日志**
   - 会话日志：查看最新的时间戳文件
   - 历史日志：查看 `nvgpu_server.log`（包含所有会话）
   - 任务日志：查看 `logs/tasks/{task_id}.*`

## 环境变量

服务器不依赖环境变量来确定路径，所有路径都基于 `main.py` 的位置动态计算。

## 迁移和部署

如果需要将 NVGPU 服务器部署到其他位置：

1. 复制整个 `nvgpu/` 目录
2. 确保保持目录结构不变
3. 更新 `configs/gpu_resource.yml` 中的 GPU 配置
4. 从新位置启动 `main.py`

所有路径会自动适配到新位置。

