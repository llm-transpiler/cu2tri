# Reasoning Content 日志设计文档

## 概述

本文档说明了如何在日志中添加 `reasoning_content` 字段，同时确保 feedback 上下文中不包含 reasoning tokens。

## 设计原则

1. **分离存储与使用**：
   - `reasoning_content` 保存在日志中用于分析
   - 发送给 LLM 的对话历史（`_conversation`）不包含 reasoning tokens
   
2. **向后兼容**：
   - 如果 `reasoning_content` 为空或 None，则不会出现在日志中
   - 现有的日志结构保持不变

## 修改的文件

### 1. `services/history.py`

#### 修改 `HistoryEvent` 数据类
```python
@dataclass
class HistoryEvent:
    role: str
    content: str
    event_type: str
    attempt_number: int
    round_id: Optional[int] = None
    retry_index: Optional[int] = None
    reasoning_content: Optional[str] = None  # 新增字段
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: ensure_timezone(now_timestamp()).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload.get("metadata"):
            payload["metadata"] = {}
        # 只在有内容时才包含 reasoning_content
        if not payload.get("reasoning_content"):
            payload.pop("reasoning_content", None)
        return payload
```

#### 修改 `add_assistant_message` 方法
```python
def add_assistant_message(
    self,
    reasoning_content: str,
    content: str,
    *,
    round_id: Optional[int],
    retry_index: Optional[int],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if self._tree is None:
        raise RuntimeError("System prompt must be set before adding assistant messages")
    # 只添加 content（不含 reasoning）到 conversation 用于 feedback 上下文
    message = make_openai_message_assistant(content)
    self._conversation.append(message)
    self._tree.add_assistant_message(text=content)
    # 在 events 中同时保存 reasoning_content 和 content 用于日志记录
    self._events.append(
        HistoryEvent(
            role="assistant",
            content=content,
            event_type="assistant",
            attempt_number=self.attempt_number,
            round_id=round_id,
            retry_index=retry_index,
            reasoning_content=reasoning_content if reasoning_content else None,
            metadata=metadata or {},
        )
    )
    self._save()
    return message
```

### 2. `services/conversation.py`

#### 修改 `save_llm_conversation` 函数签名
```python
async def save_llm_conversation(
    context: RuntimeContext,
    test_work_dir: Path,
    attempt_number: int,
    round_id: int,
    retry_index: int,
    messages: list[dict],
    full_response: str,
    *,
    model_used: str | None = None,
    reasoning_content: str | None = None,  # 新增参数
) -> None:
```

#### 在保存时包含 reasoning_content
```python
response_entry = {
    "attempt_number": attempt_number,
    "round_number": round_id,
    "retry_index": retry_index,
    "timestamp": timestamp,
    "interaction_type": "response",
    "role": "assistant",
    "content": full_response,
    "success": True,
}
if model_used:
    response_entry["model_used"] = model_used
if reasoning_content:  # 新增
    response_entry["reasoning_content"] = reasoning_content
if thinking_content:
    response_entry["thinking"] = thinking_content
    response_entry["content_without_thinking"] = re.sub(
        r"<thought>.*?</thought>|<thinking>.*?</thinking>|<think>.*?</think>",
        "",
        full_response,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
fp.write(json.dumps(response_entry, ensure_ascii=False) + "\n")
```

### 3. `services/attempts.py` 和 `services/testing.py`

在所有调用 `save_llm_conversation` 的地方添加 `reasoning_content` 参数：

```python
await save_llm_conversation(
    context=context,
    test_work_dir=attempt_work_dir,
    attempt_number=attempt_number,
    round_id=round_id,
    retry_index=retry_index,
    messages=conversation_history,
    full_response=resp_content,
    model_used=actual_model_used,
    reasoning_content=resp_reasoning_content,  # 新增
)
```

## 日志文件结构

### 1. `logs/history/conversation.json`

保存完整的 event 历史，包含 `reasoning_content`：

```json
[
  {
    "role": "system",
    "content": "You are a professional GPU computing optimization expert...",
    "event_type": "system",
    "attempt_number": 1,
    "round_id": null,
    "retry_index": null,
    "metadata": {
      "case_type": "add",
      "case_name": "add_1_15_64"
    },
    "timestamp": "2025-11-18T06:30:24.383387+08:00"
  },
  {
    "role": "user",
    "content": "# CUDA to Triton Code Conversion Task...",
    "event_type": "user",
    "attempt_number": 1,
    "round_id": 1,
    "retry_index": 0,
    "metadata": {
      "kind": "initial_prompt"
    },
    "timestamp": "2025-11-18T06:30:24.395513+08:00"
  },
  {
    "role": "assistant",
    "content": "```python\nimport torch\nimport triton...",
    "event_type": "assistant",
    "attempt_number": 1,
    "round_id": 1,
    "retry_index": 0,
    "reasoning_content": "Let me analyze the CUDA code...",
    "metadata": {
      "model_used": "openai/gpt-oss-20b",
      "usage": {
        "completion_tokens": 1000,
        "prompt_tokens": 500,
        "reasoning_tokens": 150
      }
    },
    "timestamp": "2025-11-18T06:31:54.586034+08:00"
  }
]
```

**注意**：
- 当 `reasoning_content` 为空时，该字段不会出现在 JSON 中
- `_conversation` 列表（用于 API 调用）不包含 `reasoning_content`

### 2. `logs/conversations/all_conversations.jsonl`

每行是一个 JSON 对象，response 条目包含 `reasoning_content`：

```jsonl
{"attempt_number": 1, "round_number": 1, "retry_index": 0, "timestamp": "2025-11-18T06:30:24.395513+08:00", "interaction_type": "request", "role": "system", "content": "You are a professional GPU computing optimization expert..."}
{"attempt_number": 1, "round_number": 1, "retry_index": 0, "timestamp": "2025-11-18T06:30:24.395513+08:00", "interaction_type": "request", "role": "user", "content": "# CUDA to Triton Code Conversion Task..."}
{"attempt_number": 1, "round_number": 1, "retry_index": 0, "timestamp": "2025-11-18T06:31:54.586034+08:00", "interaction_type": "response", "role": "assistant", "content": "```python\nimport torch\nimport triton...", "success": true, "model_used": "openai/gpt-oss-20b", "reasoning_content": "Let me analyze the CUDA code..."}
```

### 3. `logs/conversation_history_round_N.json`

这是发送给 LLM 的对话历史，**不包含** `reasoning_content`：

```json
[
  {
    "role": "system",
    "content": "You are a professional GPU computing optimization expert..."
  },
  {
    "role": "user",
    "content": "# CUDA to Triton Code Conversion Task..."
  },
  {
    "role": "assistant",
    "content": "```python\nimport torch\nimport triton..."
  }
]
```

**关键点**：这个文件不包含 `reasoning_content`，确保 feedback 上下文中没有 reasoning tokens。

## 数据流图

```
LLM Response
    |
    +-- resp_reasoning_content
    |       |
    |       +-- 保存到 HistoryEvent.reasoning_content
    |       |       |
    |       |       +-- logs/history/conversation.json (包含)
    |       |       +-- logs/conversations/all_conversations.jsonl (包含)
    |       |
    |       +-- 不添加到 _conversation 列表
    |
    +-- resp_content
            |
            +-- 保存到 HistoryEvent.content
            +-- 添加到 _conversation 列表
                    |
                    +-- logs/conversation_history_round_N.json (不包含 reasoning)
                    +-- 发送给 LLM 用于生成 feedback (不包含 reasoning)
```

## 使用场景

1. **分析模型推理过程**：
   - 读取 `logs/history/conversation.json` 或 `logs/conversations/all_conversations.jsonl`
   - 查看 `reasoning_content` 字段

2. **重现对话历史**：
   - 读取 `logs/conversation_history_round_N.json`
   - 这是实际发送给 LLM 的内容，不包含 reasoning tokens

3. **调试问题**：
   - 对比 `reasoning_content` 和最终的 `content`
   - 了解模型的思考过程和最终输出

## 兼容性说明

1. **向后兼容**：
   - 旧的日志文件没有 `reasoning_content` 字段，加载时会被设置为 `None`
   - 导出时如果 `reasoning_content` 为空，该字段会被自动移除

2. **未来扩展**：
   - 可以在 `extra_info` 中添加更多 reasoning 相关的元数据
   - 如 `reasoning_tokens` 数量、reasoning 时间等

## 测试建议

1. 测试有 reasoning_content 的情况
2. 测试 reasoning_content 为空的情况
3. 测试加载旧的日志文件
4. 验证发送给 LLM 的对话历史不包含 reasoning_content

