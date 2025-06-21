# OpenAI思考功能指南

本文档介绍如何在我们的LLM提供商系统中使用OpenAI的思考功能（Reasoning）。

## 概述

OpenAI的o1系列模型（包括o1、o1-mini、o3、o3-mini、o4-mini等）支持推理思考功能，这些模型可以在回答问题前进行内部推理，从而提供更准确和深思熟虑的答案。

## 功能特性

### 支持的模型
- o1、o1-mini、o1-preview
- o3、o3-mini、o3-pro
- o4-mini

### 新增参数

#### ChatRequest新增字段
```python
@dataclass
class ChatRequest:
    # ... 现有字段 ...
    
    # 思考功能配置
    include_thinking: bool = False  # 是否包含思考过程
    thinking_budget: Optional[int] = None  # 思考token预算（Google模型用）
    reasoning_effort: Optional[str] = None  # 推理强度：'low', 'medium', 'high'
```

#### ChatResponse新增字段
```python
@dataclass
class ChatResponse:
    # ... 现有字段 ...
    
    thoughts: Optional[str] = None  # 思考内容
```

#### StreamChunk新增字段
```python
@dataclass
class StreamChunk:
    # ... 现有字段 ...
    
    content_type: Optional[str] = None  # 内容类型：'thought', 'answer', 'thought_start', 'thought_end', 'error', 'end'
```

## 使用方法

### 1. 基础思考功能

```python
import asyncio
from llm.providers.impl import OpenAIProvider
from llm.providers.config import PlatformConfig
from llm.providers.base import ChatRequest, ChatMessage
from llm.providers.types import PlatformType

async def basic_example():
    config = PlatformConfig(
        platform_type=PlatformType.OPENAI_OFFICIAL,
        api_key="your-api-key"
    )
    
    provider = OpenAIProvider(config)
    
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="解释量子力学的基本原理")
        ],
        model="o1-mini",
        include_thinking=True,  # 启用思考功能
        reasoning_effort="medium",  # 设置推理强度
        max_tokens=4000
    )
    
    response = await provider.chat(request)
    
    print(f"思考信息: {response.thoughts}")
    print(f"回答: {response.content}")
    
    await provider.close()
```

### 2. 流式思考功能

```python
async def stream_example():
    config = PlatformConfig(
        platform_type=PlatformType.OPENAI_OFFICIAL,
        api_key="your-api-key"
    )
    
    provider = OpenAIProvider(config)
    
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="设计一个排序算法")
        ],
        model="o1-mini",
        include_thinking=True,
        reasoning_effort="high",
        stream=True
    )
    
    async for chunk in provider.stream_chat(request):
        print(f"[{chunk.content_type}] {chunk.content}", end="")
    
    await provider.close()
```

### 3. 不同推理强度对比

```python
async def compare_reasoning_efforts():
    provider = OpenAIProvider(config)
    
    question = "分析人工智能的发展趋势"
    
    for effort in ["low", "medium", "high"]:
        request = ChatRequest(
            messages=[ChatMessage(role="user", content=question)],
            model="o1-mini",
            include_thinking=True,
            reasoning_effort=effort
        )
        
        response = await provider.chat(request)
        print(f"推理强度 {effort}: {len(response.content)} 字符")
    
    await provider.close()
```

## API参数说明

### reasoning_effort
- **low**: 快速推理，较少的思考时间
- **medium**: 平衡推理，适中的思考时间  
- **high**: 深度推理，更多的思考时间

### 模型限制
推理模型与标准模型在参数支持上有所不同：

#### 推理模型支持的参数
- `max_completion_tokens` (而不是 `max_tokens`)
- `temperature`
- `reasoning_effort`

#### 推理模型不支持的参数
- `top_p`
- `frequency_penalty`
- `presence_penalty`

## 输出格式

### 思考信息格式
对于OpenAI模型，思考信息主要以reasoning tokens的形式提供：
```
<reasoning_tokens>1234</reasoning_tokens>
```

### 流式输出格式
```
<answer>这是模型的回答内容...</answer>
```

## 实现细节

### 模型检测
系统会自动检测是否为推理模型：
```python
def _is_reasoning_model(self, model: str) -> bool:
    reasoning_models = ['o1', 'o3', 'o4']
    return any(model.startswith(f'{m}-') or model == m for m in reasoning_models)
```

### 参数适配
系统会根据模型类型自动调整API参数：
- 推理模型：使用 `max_completion_tokens`
- 标准模型：使用 `max_tokens`

## 错误处理

常见错误及解决方案：

1. **模型不支持推理功能**
   ```
   ValidationError: Model does not support reasoning
   ```
   解决：确保使用o1、o3或o4系列模型

2. **参数不兼容**
   ```
   ProviderError: Unknown parameter for reasoning model
   ```
   解决：检查是否使用了推理模型不支持的参数

## 最佳实践

1. **选择合适的推理强度**
   - 简单问题使用 "low"
   - 复杂分析使用 "medium" 
   - 需要深度思考的问题使用 "high"

2. **设置合理的token限制**
   - 推理模型通常需要更多tokens
   - 建议设置 max_tokens >= 2000

3. **监控使用成本**
   - 推理功能会消耗额外的reasoning tokens
   - 关注 usage 信息中的 reasoning_tokens

## 示例代码

完整示例请参考：`llm/examples/openai_thinking_example.py`

## 注意事项

1. **成本考虑**: 思考功能会增加API调用成本
2. **延迟增加**: 推理过程需要更多时间
3. **模型限制**: 仅限特定的o1/o3/o4系列模型
4. **参数兼容性**: 推理模型不支持所有标准模型参数

## 与Google Provider对比

| 特性 | OpenAI | Google |
|------|--------|--------|
| 思考内容可见性 | 仅reasoning tokens数量 | 完整思考过程 |
| 流式思考 | 不支持真实流式思考 | 支持思考过程流式输出 |
| 参数控制 | reasoning_effort | thinking_budget |
| 输出格式 | 简化格式 | 详细的thought标签 | 