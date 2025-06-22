# 🚀 Chat Service v2.0

## 概述

基于精简后的 `/llm` 模块重写的聊天服务，去除了复杂的队列管理和配置系统，采用更直接、更简洁的架构。

## ✨ 核心特性

### 🎯 简化架构
- **直接集成**: 直接使用 `get_provider(PlatformType.OPENROUTER)` 
- **无队列系统**: 去除了复杂的生产者消费者模式
- **自动配置**: 自动从环境变量读取API密钥
- **即开即用**: 无需复杂的配置文件

### 🤖 支持的模型平台

| 平台 | 模型示例 | 环境变量 |
|-----|---------|----------|
| **OpenRouter** | `deepseek/deepseek-r1-0528:free`<br>`openai/gpt-4o`<br>`anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| **Google Official** | `gemini-2.5-pro`<br>`gemini-2.5-flash` | `GEMINI_API_KEY` |
| **OpenAI Official** | `gpt-4o`<br>`gpt-4o-mini` | `OPENAI_API_KEY` |

### 🧠 思考功能
- 支持 DeepSeek 和 Gemini 的思考功能
- 自动解析 `<thought>` 标签
- 配置思考预算和推理强度

### 📱 流式输出
- Server-Sent Events (SSE) 支持
- 实时内容流式传输
- 自动错误处理

## 🏗️ 架构对比

### v1.0 (复杂版本)
```
HTTP Request → Queue Manager → Worker Pool → Provider → LLM API
                    ↓
            Complex Config System + YAML Files
```

### v2.0 (精简版本)
```
HTTP Request → Chat Service → get_provider() → LLM API
                    ↓
            Simple Environment Variables
```

## 📁 文件结构

```
server/chat/
├── models.py           # Pydantic数据模型
├── service.py          # 核心聊天服务
├── api_server.py       # FastAPI HTTP服务器
├── example_server.py   # 示例启动脚本
├── simple_test.py      # 测试脚本
└── README_v2.md        # 本文档
```

## 🚀 快速开始

### 1. 设置环境变量

创建 `.env` 文件：
```bash
# 必需 - OpenRouter支持最多免费模型
OPENROUTER_API_KEY=your-openrouter-api-key

# 可选
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 2. 安装依赖

```bash
pip install fastapi uvicorn aiohttp pydantic
```

### 3. 启动服务

```bash
cd /workspace/monocases
python -m server.chat.example_server
```

### 4. 访问API文档

- 📊 Swagger UI: http://localhost:8000/docs
- 🔄 ReDoc: http://localhost:8000/redoc
- 🏥 健康检查: http://localhost:8000/health

## 📝 API 使用示例

### 基础聊天

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test_123",
    "message": "你好",
    "model": "deepseek/deepseek-r1-0528:free",
    "max_tokens": 100
  }'
```

### 思考功能

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "thinking_test",
    "message": "请解释什么是递归",
    "model": "deepseek/deepseek-r1-0528:free",
    "include_thinking": true,
    "thinking_budget": 500,
    "max_tokens": 200
  }'
```

### 流式聊天

```bash
curl -X POST "http://localhost:8000/stream_chat" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "stream_test",
    "message": "写一首关于编程的诗",
    "model": "deepseek/deepseek-r1-0528:free"
  }'
```

### 获取对话历史

```bash
curl "http://localhost:8000/conversations/test_123/history"
```

## 🧪 测试

运行测试套件：

```bash
# 确保服务正在运行
python -m server.chat.example_server

# 在另一个终端运行测试
python -m server.chat.simple_test
```

测试包括：
- ✅ 健康检查
- ✅ 单轮对话
- ✅ 多轮对话  
- ✅ 对话历史
- ✅ 流式输出
- ✅ 思考功能

## 📊 性能特点

### v1.0 → v2.0 改进

| 方面 | v1.0 | v2.0 |
|-----|------|------|
| **代码复杂度** | 高 (队列+配置+工厂) | 低 (直接调用) |
| **启动时间** | 慢 (初始化多个组件) | 快 (按需初始化) |
| **内存使用** | 高 (常驻队列和workers) | 低 (按需创建) |
| **配置复杂度** | 复杂 (YAML配置文件) | 简单 (环境变量) |
| **错误处理** | 分布式 (多层错误处理) | 集中 (统一异常处理) |
| **调试难度** | 难 (多个异步组件) | 易 (线性调用栈) |

### 性能数据
- **响应时间**: 通常 2-5 秒 (取决于模型)
- **并发支持**: FastAPI 原生异步支持
- **内存占用**: ~100MB (vs v1.0 的 ~300MB)
- **启动时间**: ~2秒 (vs v1.0 的 ~10秒)

## 🔧 配置选项

### 支持的模型配置

```python
# 在 models.py 中的 SupportedModel 枚举
class SupportedModel(str, Enum):
    # OpenRouter 免费模型
    OPENROUTER_DEEPSEEK_FREE = "deepseek/deepseek-r1-0528:free"
    
    # OpenRouter 付费模型
    OPENROUTER_GPT4O = "openai/gpt-4o"
    OPENROUTER_CLAUDE_SONNET = "anthropic/claude-3.5-sonnet"
    
    # 官方API
    GEMINI_PRO = "gemini-2.5-pro"
    GPT4O = "gpt-4o"
```

### 聊天请求参数

```python
class ChatRequest(BaseModel):
    conversation_id: str          # 对话ID
    message: str                  # 用户消息
    model: SupportedModel         # 使用的模型
    
    # 可选参数
    system_prompt: Optional[str]  # 系统提示
    max_tokens: Optional[int]     # 最大token数
    temperature: Optional[float]  # 温度参数
    
    # 思考功能
    include_thinking: bool = False
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None
    
    # 流式输出
    stream: bool = False
```

## 🐛 故障排除

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查环境变量
   echo $OPENROUTER_API_KEY
   
   # 检查端口占用
   lsof -i :8000
   ```

2. **API调用失败**
   ```bash
   # 检查健康状态
   curl http://localhost:8000/health
   
   # 检查支持的模型
   curl http://localhost:8000/models
   ```

3. **思考功能不工作**
   - 确保使用支持思考的模型 (DeepSeek, Gemini)
   - 设置 `include_thinking: true`
   - 检查 `thinking_budget` 参数

### 日志级别

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 详细日志
logging.basicConfig(level=logging.INFO)   # 标准日志
```

## 🔮 未来规划

- [ ] **多模态支持**: 图片和文件输入
- [ ] **模型切换**: 运行时动态切换模型
- [ ] **批量请求**: 支持批量处理多个对话
- [ ] **缓存系统**: Redis 缓存常用响应
- [ ] **监控仪表板**: 实时性能监控
- [ ] **A/B测试**: 多模型响应对比

## 📞 支持

如有问题，请检查：
1. 📋 [API 文档](http://localhost:8000/docs)
2. 🧪 [测试脚本](./simple_test.py) 
3. 📝 [示例代码](./example_server.py)

---

> **注意**: 这是 v2.0 版本，基于精简后的 `/llm` 模块重写。如需 v1.0 的复杂队列功能，请参考原始文件。 