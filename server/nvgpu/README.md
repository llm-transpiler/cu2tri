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

```shell
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
│   ├── API_REFERENCE_ZH.md  # 中文API参考手册
│   ├── API_REFERENCE.md     # 英文API参考手册
│   ├── README.md            # 文档中心
│   ├── QUICKSTART.md        # 快速入门指南
│   ├── DESIGN.md            # 完整设计文档
│   ├── CHANGES.md           # 版本变更说明
│   ├── CHANGELOG.md         # 变更日志
│   ├── GPU_CONFIG_GUIDE.md  # GPU配置指南
│   ├── WORK_DIR_GUIDE.md    # 工作目录指南
│   ├── DIRECTORY_STRUCTURE.md  # 目录结构说明
│   └── TODO.md              # 待办事项
│
├── scripts/                 # 工具脚本
│   ├── test_api.sh          # API测试脚本
│   ├── test_gpu_mapping.py  # GPU映射测试
│   └── example_test_script.py  # 示例测试脚本
│
├── examples/                # 客户端使用示例
│   ├── README.md            # 示例说明文档
│   ├── example_basic.py     # 基础使用示例
│   ├── example_batch_submit.py  # 批量提交示例
│   ├── example_concurrent_tasks.py  # 并发任务示例
│   ├── example_custom_env.py  # 自定义环境变量示例
│   ├── example_error_handling.py  # 错误处理示例
│   ├── example_gpu_management.py  # GPU管理示例
│   ├── example_log_handling.py  # 日志处理示例
│   ├── example_mode_switching.py  # GPU模式切换示例
│   ├── example_cancel_running_task.py  # 任务取消示例
│   ├── example_task_parameters.py  # 任务参数示例（完整）
│   ├── example_task_parameters_quick.py  # 任务参数示例（快速）
│   └── README_TASK_PARAMETERS.md  # 任务参数说明
│
├── test_scripts/            # 测试用脚本
│   ├── simple_functional_test.py
│   ├── failing_test.py
│   ├── long_running_task.py
│   └── gpu_memory_intensive.py
│
├── tests/                   # 单元测试（108个测试用例，100%通过）
│   ├── README.md            # 测试套件说明
│   ├── conftest.py          # Pytest fixtures
│   ├── pytest.ini           # Pytest配置
│   ├── requirements-test.txt  # 测试依赖
│   ├── run_tests.sh         # 测试运行脚本
│   ├── test_models.py       # 数据模型测试（15个）
│   ├── test_task_queue.py   # 任务队列测试（24个）
│   ├── test_gpu_manager.py  # GPU管理器测试（23个）
│   ├── test_scheduler.py    # 调度器测试（12个）
│   ├── test_client.py       # 客户端测试（17个）
│   ├── test_integration.py  # 集成测试（17个）
│   └── TEST_COVERAGE.md     # 测试覆盖率报告
│
└── logs/                    # 日志目录（运行时创建）
    ├── nvgpu_server.log     # 历史日志（追加）
    ├── nvgpu_server_YYYYMMDD_HHMMSS.log  # 会话日志（时间戳）
    └── tasks/               # 任务日志

```

## 文档

### 📚 核心文档
- **[快速入门](docs/QUICKSTART.md)** ⭐⭐⭐ - 5分钟上手指南
- **[文档中心](docs/README.md)** - 完整文档索引
- **[设计文档](docs/DESIGN.md)** ⭐⭐⭐⭐ - 完整设计说明（强烈推荐）
- **[变更说明](docs/CHANGES.md)** ⭐⭐⭐ - 版本变更详情
- **[API参考手册](docs/API_REFERENCE.md)** ⭐⭐⭐ - 完整的API文档

### 🔧 配置与参考
- **[GPU配置指南](docs/GPU_CONFIG_GUIDE.md)** - 如何配置GPU资源
- **[工作目录指南](docs/WORK_DIR_GUIDE.md)** - work_dir功能说明
- **[目录结构](docs/DIRECTORY_STRUCTURE.md)** - 文件组织和路径说明

### 📋 其他
- **[变更日志](docs/CHANGELOG.md)** - 版本更新记录
- **[待办事项](docs/TODO.md)** - 开发计划

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

