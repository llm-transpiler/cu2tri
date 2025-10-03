# NVGPU Server

GPU任务调度服务器，支持多GPU管理、任务队列、性能测试等功能。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务器
python main.py

# 使用自定义配置
python main.py --gpu-config configs/gpu_resource.yml
```

## 目录结构

```
nvgpu/
├── README.md                # 本文件
├── requirements.txt         # Python依赖
├── main.py                  # 服务器入口
├── config.py                # 配置管理
├── client.py                # Python客户端库
├── models.py                # 数据模型
├── api_server.py            # REST API服务
├── gpu_manager.py           # GPU管理器
├── task_queue.py            # 任务队列
├── task_runner.py           # 任务执行器
├── scheduler.py             # 调度器
├── logger.py                # 日志系统
├── gpu_config_loader.py     # GPU配置加载器
│
├── configs/                 # 配置文件
│   └── gpu_resource.yml     # GPU资源配置
│
├── docs/                    # 文档
│   ├── QUICKSTART.md        # 快速入门指南
│   ├── README.md            # 详细说明文档
│   ├── GPU_CONFIG_GUIDE.md  # GPU配置指南
│   ├── LOG_HANDLING.md      # 日志处理文档
│   ├── TASK_CANCELLATION.md # 任务取消文档
│   ├── MAX_CONCURRENT_TASKS.md  # 并发任务限制
│   ├── CHANGELOG.md         # 变更日志
│   ├── CHANGELOG_v1.2.md    # v1.2版本变更
│   ├── UPDATE_NOTES.md      # 更新说明
│   ├── UPDATES_v1.3.md      # v1.3版本更新
│   └── PROJECT_SUMMARY.md   # 项目总结
│
├── scripts/                 # 工具脚本
│   ├── test_api.sh          # API测试脚本
│   ├── test_gpu_mapping.py  # GPU映射测试
│   └── example_test_script.py  # 示例测试脚本
│
├── examples/                # 客户端使用示例
│   ├── example_basic.py
│   ├── example_batch_submit.py
│   ├── example_concurrent_tasks.py
│   ├── example_custom_env.py
│   ├── example_error_handling.py
│   ├── example_gpu_management.py
│   ├── example_log_handling.py
│   └── example_cancel_running_task.py
│
├── test_scripts/            # 测试用脚本
│   ├── simple_functional_test.py
│   ├── failing_test.py
│   ├── long_running_task.py
│   └── gpu_memory_intensive.py
│
└── logs/                    # 日志目录（运行时创建）
    ├── nvgpu_server.log     # 历史日志（追加）
    ├── nvgpu_server_YYYYMMDD_HHMMSS.log  # 会话日志（时间戳）
    └── tasks/               # 任务日志

```

## 文档

- **[快速入门](docs/QUICKSTART.md)** - 5分钟上手指南
- **[详细文档](docs/README.md)** - 完整功能说明
- **[目录结构](docs/DIRECTORY_STRUCTURE.md)** - 文件组织和路径说明
- **[GPU配置指南](docs/GPU_CONFIG_GUIDE.md)** - 如何配置GPU资源
- **[变更日志](docs/CHANGELOG.md)** - 版本更新记录

## API文档

服务器启动后，访问以下端点：

- `GET /health` - 健康检查
- `POST /tasks` - 提交任务
- `GET /tasks/{task_id}` - 查询任务状态
- `POST /tasks/{task_id}/cancel` - 取消任务
- `GET /gpus` - 列出所有GPU
- `GET /stats` - 服务器统计信息

详见 [docs/README.md](docs/README.md)

## Python客户端

```python
from server.nvgpu.client import NVGPUClient

client = NVGPUClient("http://localhost:8080")

# 提交任务
task_id = client.submit_task(
    script_path="test.py",
    task_type="functional"
)

# 查询状态
result = client.get_task(task_id)
print(result.status)
```

更多示例见 [examples/](examples/) 目录。

## 许可证

[待定]

