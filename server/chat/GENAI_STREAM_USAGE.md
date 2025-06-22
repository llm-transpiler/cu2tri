# Genai 流式聊天功能使用指南

## 概述

本指南介绍如何在聊天服务中使用 Gemini API (genai) 的流式聊天功能。新增的功能支持实时流式输出，同时保持与现有队列系统的兼容性。

## 功能特性

### ✨ **核心特性**
- 🚀 **流式输出捕获**：所有 Gemini API 调用默认使用流式输出捕获
- 📡 **实时响应**：支持服务器推送事件 (SSE) 格式的实时输出
- 🔄 **对话历史**：自动管理多轮对话历史记录
- 🎯 **专用端点**：为 Gemini 模型提供专门的流式接口
- ⚡ **性能优化**：比传统 API 调用更快的响应速度

### 🛠 **技术特性**
- 异步生成器支持
- 错误处理和恢复
- Thought 标签解析
- 多模型支持（genai + OpenRouter）

## API 端点

### 1. 通用流式聊天接口
```
POST /stream_chat
```

支持所有提供商的流式聊天（包括 genai 和 OpenRouter）。

**请求示例：**
```json
{
  "conversation_id": "test_conversation",
  "message": "写一个Python Hello World程序",
  "model": "gemini-2.5-pro-preview-06-05",
  "temperature": 0.7,
  "stream": true
}
```

**响应格式（SSE）：**
```
data: {"content": "当然！以下是一个简单的Python Hello World程序："}

data: {"content": "\n\n```python\nprint(\"Hello, World!\")\n```"}

data: {"finish_reason": "stop"}

data: [DONE]
```

### 2. Gemini 专用流式接口
```
POST /genai_stream_chat
```

专门为 Gemini 模型优化的流式接口，提供更详细的响应信息。

**请求示例：**
```json
{
  "conversation_id": "genai_test",
  "message": "解释一下Python装饰器",
  "model": "gemini-2.0-flash",
  "temperature": 0.7
}
```

**响应格式（SSE）：**
```
data: {"content": "Python装饰器是一种", "model": "gemini-2.0-flash", "chunk_id": 1, "finish_reason": null}

data: {"content": "设计模式，允许你", "model": "gemini-2.0-flash", "chunk_id": 2, "finish_reason": null}

data: {"content": "", "model": "gemini-2.0-flash", "chunk_id": 3, "finish_reason": "stop"}

data: [DONE]
```

## 编程接口使用

### 1. 基本流式聊天

```python
import asyncio
from server.chat.service import ChatService
from server.chat.models import ChatRequest, ModelName

async def basic_stream_example():
    # 创建聊天服务
    chat_service = ChatService(model_configs=your_configs)
    await chat_service.start()
    
    # 创建请求
    request = ChatRequest(
        conversation_id="example_conversation",
        message="写一个Python函数计算斐波那契数列",
        model=ModelName.GENAI_GEMINI_2_5_PRO,
        stream=True
    )
    
    # 流式响应
    print("AI响应：", end="")
    async for chunk in chat_service.stream_chat(request):
        print(chunk, end="", flush=True)
    print()  # 换行
    
    await chat_service.stop()

# 运行示例
asyncio.run(basic_stream_example())
```

### 2. 多轮对话示例

```python
async def conversation_example():
    chat_service = ChatService(model_configs=your_configs)
    await chat_service.start()
    
    conversation_id = "multi_turn_chat"
    
    # 第一轮对话
    request1 = ChatRequest(
        conversation_id=conversation_id,
        message="你好，我想学习Python",
        model=ModelName.GENAI_GEMINI_2_0_FLASH
    )
    
    response1_parts = []
    async for chunk in chat_service.stream_chat(request1):
        response1_parts.append(chunk)
        print(chunk, end="", flush=True)
    print("\n" + "="*50)
    
    # 第二轮对话（会自动包含历史）
    request2 = ChatRequest(
        conversation_id=conversation_id,
        message="从哪里开始比较好？",
        model=ModelName.GENAI_GEMINI_2_0_FLASH
    )
    
    async for chunk in chat_service.stream_chat(request2):
        print(chunk, end="", flush=True)
    print()
    
    await chat_service.stop()
```

### 3. HTTP 客户端示例

```python
import requests
import json

def http_stream_example():
    url = "http://localhost:8000/genai_stream_chat"
    
    data = {
        "conversation_id": "http_test",
        "message": "用Python写一个简单的web服务器",
        "model": "gemini-2.5-pro-preview-06-05",
        "temperature": 0.7
    }
    
    # 发送流式请求
    response = requests.post(url, json=data, stream=True)
    
    print("流式响应：")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]  # 移除 'data: ' 前缀
                if data_str != '[DONE]':
                    try:
                        data_obj = json.loads(data_str)
                        if 'content' in data_obj and data_obj['content']:
                            print(data_obj['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        pass
    print()
```

## 配置示例

### 1. 模型配置

```python
from server.chat.models import ModelConfig, ModelProvider

model_configs = {
    # Gemini 模型配置（通过 genai）
    "gemini-2.5-pro-preview-06-05": ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-2.5-pro-preview-06-05",
        api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=4096,
        temperature=0.7
    ),
    
    "gemini-2.0-flash": ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=4096,
        temperature=0.7
    ),
    
    # OpenRouter 配置（作为对比）
    "google/gemini-2.5-pro": ModelConfig(
        provider=ModelProvider.OPENAI,  # 通过 OpenRouter
        model_name="google/gemini-2.5-pro",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=4096,
        temperature=0.7
    )
}
```

### 2. 服务启动配置

```python
import asyncio
import logging
from server.chat.api_server import create_chat_api_server

async def start_server():
    # 创建API服务器
    api_server = create_chat_api_server(
        model_configs=model_configs,
        host="0.0.0.0",
        port=8000,
        max_workers=10,
        logger=logging.getLogger(__name__)
    )
    
    # 启动服务器
    await api_server.start()

# 启动服务
asyncio.run(start_server())
```

## 性能优势

### 📊 **性能对比**

| 特性 | Genai 直接调用 | OpenRouter |
|------|----------------|------------|
| 响应延迟 | ⚡ 较低 | 🐌 较高 |
| 流式支持 | ✅ 原生支持 | ✅ 支持 |
| 错误处理 | 🛡️ 完善 | 🛡️ 完善 |
| 并发支持 | 🚀 优秀 | 🚀 良好 |

### 💡 **最佳实践**

1. **选择合适的模型**：
   - `gemini-2.0-flash`：快速响应，适合简单任务
   - `gemini-2.5-pro`：复杂推理，适合难题

2. **优化参数配置**：
   - `temperature=0.7`：平衡创造性和准确性
   - `max_tokens=4096`：根据需要调整

3. **错误处理**：
   ```python
   try:
       async for chunk in chat_service.stream_chat(request):
           print(chunk, end="", flush=True)
   except Exception as e:
       print(f"流式响应错误: {e}")
   ```

## 环境要求

### 📋 **必需依赖**
```bash
pip install google-genai fastapi uvicorn aiohttp
```

### 🔑 **环境变量**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export OPENROUTER_API_KEY="your_openrouter_key_here"  # 可选
```

## 测试验证

运行测试套件验证功能：

```bash
# 基本功能测试
python server/chat/test_genai_stream.py

# 流式输出捕获测试
python cu2tri/test_stream_capture.py
```

## 故障排除

### 🔧 **常见问题**

1. **API Key 未设置**
   ```
   Error: GEMINI_API_KEY is required
   ```
   **解决方案**：设置正确的环境变量

2. **模型不支持**
   ```
   Error: Model not supported
   ```
   **解决方案**：检查模型名称是否在 ModelName 枚举中

3. **流式响应中断**
   ```
   Error: Stream generation error
   ```
   **解决方案**：检查网络连接和 API 配额

### 📝 **调试技巧**

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 总结

Genai 流式聊天功能为聊天服务提供了：

- ✅ **高性能**：更快的响应速度
- ✅ **实时性**：流式输出体验
- ✅ **兼容性**：与现有系统无缝集成
- ✅ **稳定性**：完善的错误处理机制

通过本指南，您可以轻松集成和使用 Gemini API 的流式聊天功能，为用户提供更好的交互体验。 