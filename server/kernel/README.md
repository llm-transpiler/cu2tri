# 内核开发服务器系统

## 🚀 系统概述

这是一个完整的CUDA/Triton内核开发、测试和优化服务器系统，支持：
- 自动化内核编译
- 功能正确性测试
- 性能基准测试
- LLM辅助的多轮优化
- 形式化验证
- 智能GPU资源管理

## 🏗️ 系统架构

```
server/kernel/
├── __init__.py              # 包初始化
├── models.py                # 数据模型定义
├── gpu_manager.py           # GPU资源管理器
├── queue_manager.py         # 任务队列管理器
├── service.py               # 主服务逻辑
├── api_server.py            # REST API服务器
├── processors/              # 任务处理器模块
│   ├── __init__.py
│   ├── base.py             # 基础处理器
│   ├── compiler.py         # 编译器
│   ├── tester.py           # 功能测试器
│   ├── performance.py      # 性能测试器
│   ├── llm_processor.py    # LLM处理器
│   ├── rag_processor.py    # RAG检索器
│   ├── web_search.py       # Web搜索器
│   └── smt_verifier.py     # SMT验证器
├── tests/                   # 测试脚本
├── config/                  # 配置文件
└── logs/                    # 日志文件
```

## 🔧 安装与配置

### 1. 环境依赖

```bash
# 基础依赖
pip install fastapi uvicorn pydantic numpy torch
pip install nvidia-ml-py3 psutil requests beautifulsoup4
pip install chromadb sentence-transformers
pip install z3-solver  # SMT求解器

# GPU监控工具
sudo apt-get install nvidia-smi
```

### 2. 环境变量配置

创建 `.env` 文件：

```bash
# LLM Provider配置
OPENROUTER_API_KEY=your_openrouter_key
GEMINI_API_KEY=your_gemini_key

# 服务配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO

# GPU配置
DEV_GPU_IDS=0,1        # L20 GPUs for development
PROD_GPU_IDS=2,3,4,5   # H100 GPUs for production
```

## 🎯 核心功能

### 1. GPU资源管理

```python
from server.kernel.gpu_manager import GPUManager

# 初始化GPU管理器
gpu_manager = GPUManager()

# 分配GPU资源
gpu_id = await gpu_manager.allocate_gpu(stage='development')
print(f"分配到GPU: {gpu_id}")

# 释放GPU资源
await gpu_manager.release_gpu(gpu_id)
```

### 2. 内核编译

```python
from server.kernel.models import CompileRequest, KernelType

# 创建编译请求
compile_req = CompileRequest(
    kernel_type=KernelType.TRITON,
    source_code="""
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
""",
    compile_options={"optimize": True}
)

# 提交编译任务
result = await service.compile_kernel(compile_req)
```

### 3. 功能测试

```python
from server.kernel.models import TestRequest

# 创建测试请求
test_req = TestRequest(
    kernel_name="add_kernel",
    test_inputs=[
        {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]},
        {"x": [10, 20], "y": [30, 40]}
    ],
    expected_outputs=[
        [6, 8, 10, 12],
        [40, 60]
    ]
)

# 执行功能测试
result = await service.test_kernel(test_req)
```

### 4. 性能测试

```python
from server.kernel.models import PerfRequest

# 创建性能测试请求
perf_req = PerfRequest(
    kernel_name="add_kernel",
    input_sizes=[(1024,), (4096,), (16384,)],
    num_runs=100,
    warmup_runs=10
)

# 执行性能测试
result = await service.benchmark_kernel(perf_req)
print(f"平均执行时间: {result.avg_time_ms}ms")
print(f"GPU利用率: {result.gpu_utilization}%")
```

### 5. LLM辅助优化

```python
from server.kernel.models import LLMRequest

# 创建LLM优化请求
llm_req = LLMRequest(
    conversation_id="cuda_to_triton_001",
    prompt="请将以下CUDA内核转换为Triton内核",
    context={
        "cuda_code": cuda_kernel_code,
        "performance_data": perf_results
    }
)

# 获取LLM建议
result = await service.llm_assist(llm_req)
```

## 🚦 快速开始

### 1. 启动服务器

```bash
# 进入项目目录
cd /workspace/monocases/server/kernel

# 启动API服务器
python api_server.py
```

### 2. 测试基本功能

```bash
# 运行基础测试
python tests/test_basic.py

# 运行GPU特定测试（使用GPU 5）
python tests/test_gpu5.py

# 运行完整集成测试
python tests/test_integration.py
```

### 3. 使用REST API

```bash
# 检查服务器状态
curl http://localhost:8000/health

# 提交编译任务
curl -X POST http://localhost:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"kernel_type": "triton", "source_code": "..."}'

# 查看任务状态
curl http://localhost:8000/task/{task_id}
```

## 📊 监控与日志

### 1. 实时监控

```python
# 查看GPU状态
from server.kernel.gpu_manager import GPUManager
gpu_manager = GPUManager()
stats = await gpu_manager.get_gpu_stats()
print(stats)

# 查看队列状态
from server.kernel.queue_manager import QueueManager
queue_manager = QueueManager()
queue_stats = queue_manager.get_queue_stats()
print(queue_stats)
```

### 2. 日志系统

日志文件位置：
- `logs/kernel_service.log` - 主服务日志
- `logs/gpu_manager.log` - GPU管理日志
- `logs/queue_manager.log` - 队列管理日志

## 🔗 API接口文档

### 编译接口
- `POST /compile` - 提交编译任务
- `GET /compile/{task_id}` - 查看编译结果

### 测试接口
- `POST /test` - 提交功能测试
- `POST /benchmark` - 提交性能测试

### LLM接口
- `POST /llm/assist` - LLM辅助优化
- `POST /llm/conversation` - 多轮对话

### 系统接口
- `GET /health` - 系统健康检查
- `GET /gpu/status` - GPU状态查询
- `GET /queue/stats` - 队列统计信息

## 🛠️ 高级配置

### 1. GPU分配策略

```python
# 自定义GPU分配策略
GPU_ALLOCATION_CONFIG = {
    'development': {
        'gpu_ids': [0, 1],  # L20 GPUs
        'max_concurrent': 2,
        'priority': 'low'
    },
    'production': {
        'gpu_ids': [2, 3, 4, 5],  # H100 GPUs
        'max_concurrent': 4,
        'priority': 'high'
    }
}
```

### 2. 性能调优

```python
# 队列管理器配置
QUEUE_CONFIG = {
    'max_workers': 8,
    'retry_attempts': 3,
    'timeout_seconds': 300,
    'priority_levels': 5
}

# 编译器配置
COMPILER_CONFIG = {
    'optimization_level': 'O3',
    'enable_profiling': True,
    'cache_compiled_kernels': True
}
```

## 🐛 故障排除

### 常见问题

1. **GPU不可用**
   ```bash
   # 检查GPU状态
   nvidia-smi
   
   # 检查CUDA安装
   nvcc --version
   ```

2. **编译失败**
   ```bash
   # 检查编译环境
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import triton; print(triton.__version__)"
   ```

3. **服务启动失败**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 查看详细日志
   tail -f logs/kernel_service.log
   ```

## 📈 性能优化建议

1. **GPU资源优化**
   - 合理分配开发/生产GPU
   - 监控GPU利用率
   - 使用GPU池化技术

2. **内存管理**
   - 及时释放GPU内存
   - 使用内存池
   - 监控内存使用情况

3. **任务调度优化**
   - 基于优先级的调度
   - 负载均衡
   - 批处理任务

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送分支
5. 创建Pull Request

## 📄 许可证

MIT License

## 🆘 支持

如有问题，请：
1. 查看日志文件
2. 检查GPU状态
3. 提交Issue
4. 联系维护者

---

**版本**: 1.0.0  
**最后更新**: 2024-01-01  
**维护者**: AI Assistant 