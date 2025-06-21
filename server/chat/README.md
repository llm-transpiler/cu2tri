# 大模型聊天服务

基于生产者消费者模式的高性能大模型聊天服务，支持多种模型提供商和多轮对话。

## 🌟 特性

- **生产者消费者模式**: 使用异步队列处理请求，支持高并发
- **多模型支持**: 支持 OpenAI、Claude、DeepSeek、Gemini 等多种模型
- **多轮对话**: 完整的对话历史管理，支持上下文记忆
- **思考过程**: 支持记录和处理模型的思考过程 (Thought)
- **高性能**: 协程并发处理，支持优先级队列
- **监控统计**: 实时性能监控和统计信息
- **REST API**: 完整的HTTP API接口
- **压力测试**: 内置压力测试工具

## 🏗️ 架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTTP API      │───▶│   Queue Manager  │───▶│   Chat Service  │
│   (FastAPI)     │    │   (Producer/     │    │   (Providers)   │
└─────────────────┘    │    Consumer)     │    └─────────────────┘
                       └──────────────────┘              │
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Conversation   │    │   API Providers │
                       │    Manager       │    │   - OpenRouter  │
                       │   (LLM History)  │    │   - Gemini      │
                       └──────────────────┘    └─────────────────┘
```

## 📦 安装依赖

```bash
# 基础依赖
pip install fastapi uvicorn aiohttp openai

# Gemini支持 (可选)
pip install google-genai

# 测试工具
pip install pytest pytest-asyncio
```

## 🚀 快速开始

### 1. 设置环境变量

```bash
# OpenRouter API密钥 (支持 OpenAI、Claude、DeepSeek)
export OPENROUTER_API_KEY=your-openrouter-api-key

# Gemini API密钥 (可选)
export GEMINI_API_KEY=your-gemini-api-key
```

### 2. 启动服务

```bash
# 启动聊天服务
python -m server.chat.example_server
```

服务将在 `http://localhost:8000` 启动。

### 3. 访问API文档

打开浏览器访问 `http://localhost:8000/docs` 查看自动生成的API文档。

## 📚 API 接口

### 聊天接口

```http
POST /chat
Content-Type: application/json

{
    "conversation_id": "conv_123",
    "message": "Hello, how are you?",
    "model": "openai/gpt-4o-mini",
    "thought": "This is a test message",
    "max_tokens": 100,
    "temperature": 0.7
}
```

### 获取对话历史

```http
GET /conversations/{conversation_id}/history?model=openai/gpt-4o-mini
```

### 服务统计

```http
GET /stats
```

### 健康检查

```http
GET /health
```

## 🎯 支持的模型

### OpenAI (通过 OpenRouter)
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

### Claude (通过 OpenRouter)
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-haiku`

### DeepSeek (通过 OpenRouter)
- `deepseek/deepseek-r1`
- `deepseek/deepseek-chat`

### Gemini (直接调用或通过 OpenRouter)
- `google/gemini-pro`
- `google/gemini-flash-1.5`

## 🧪 测试

### 基本功能测试

```bash
python -m server.chat.simple_test
```

### 压力测试

```bash
# 并发测试
python -m server.chat.stress_test --test concurrent --requests 100 --concurrency 10

# 负载测试
python -m server.chat.stress_test --test load --duration 60 --rps 10

# 多轮对话测试
python -m server.chat.stress_test --test conversation --conversations 5

# 完整测试
python -m server.chat.stress_test --test all
```

### 测试参数

- `--url`: 服务器URL (默认: http://localhost:8000)
- `--requests`: 并发测试总请求数 (默认: 100)
- `--concurrency`: 并发数 (默认: 10)
- `--duration`: 负载测试持续时间，秒 (默认: 60)
- `--rps`: 负载测试每秒请求数 (默认: 10)
- `--conversations`: 对话数量 (默认: 5)

## ⚙️ 配置选项

### 服务配置

```python
from server.chat.service import ChatService
from server.chat.models import ModelConfig, ModelProvider

# 创建模型配置
model_configs = {
    "openai/gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="openai/gpt-4o-mini",
        api_key="your-api-key",
        max_tokens=4096,
        temperature=0.7,
        timeout=30
    )
}

# 创建聊天服务
chat_service = ChatService(
    model_configs=model_configs,
    max_workers=20,        # 最大工作协程数
    max_queue_size=1000,   # 最大队列大小
)
```

### 队列配置

```python
from server.chat.queue_manager import ChatQueueManager

queue_manager = ChatQueueManager(
    max_workers=10,        # 工作协程数
    max_queue_size=1000,   # 队列大小
    task_timeout=300,      # 任务超时时间(秒)
)
```

## 📊 性能监控

服务提供实时性能监控：

```python
# 获取统计信息
stats = chat_service.get_stats()

print(f"总请求数: {stats['queue']['total_requests']}")
print(f"成功率: {stats['queue']['completed_requests'] / stats['queue']['total_requests'] * 100:.2f}%")
print(f"平均响应时间: {stats['queue']['average_processing_time']:.3f}秒")
```

## 🔧 高级用法

### 自定义提供商

```python
from server.chat.providers import ChatProvider

class CustomProvider(ChatProvider):
    async def chat(self, request, history):
        # 实现自定义聊天逻辑
        pass
    
    async def stream_chat(self, request, history):
        # 实现流式响应
        pass
```

### 优先级队列

```python
# 高优先级请求 (数字越小优先级越高)
response = await chat_service.chat(request, priority=0)

# 普通请求
response = await chat_service.chat(request, priority=5)
```

### 思考过程处理

```python
# 发送带思考的请求
request = ChatRequest(
    conversation_id="conv_123",
    message="Solve this math problem: 2+2",
    model=ModelName.GPT_4O_MINI,
    thought="I need to carefully solve this step by step"
)

response = await chat_service.chat(request)
print(f"Assistant's thought: {response.thought}")
print(f"Assistant's answer: {response.message}")
```

## 🐛 故障排除

### 常见问题

1. **连接失败**: 检查服务是否启动，端口是否被占用
2. **API密钥错误**: 确认环境变量设置正确
3. **模型不支持**: 检查模型名称是否正确
4. **请求超时**: 调整timeout配置或检查网络连接

### 日志调试

```python
import logging

# 启用调试日志
logging.basicConfig(level=logging.DEBUG)
```

## 📈 性能优化建议

1. **调整工作协程数**: 根据服务器性能调整 `max_workers`
2. **优化队列大小**: 合理设置 `max_queue_size` 避免内存溢出
3. **使用连接池**: HTTP客户端使用连接池减少连接开销
4. **监控资源使用**: 定期检查CPU、内存、网络使用情况
5. **缓存策略**: 对重复请求实现缓存机制

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Web框架
- [OpenRouter](https://openrouter.ai/) - 统一的AI模型API
- [Google Gemini](https://ai.google.dev/) - Google的AI模型
- [OpenAI](https://openai.com/) - GPT模型提供商 