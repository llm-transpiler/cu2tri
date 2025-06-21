# LLM History 模块

## 概述

LLM History 模块提供了树形结构的对话历史管理功能。它允许用户构建和管理复杂的对话树，支持分支对话、历史回溯、节点切换等高级功能。

## 核心概念

### 树形对话结构

```
系统消息 (root)
├── 用户消息 1
│   ├── 助手回复 1a
│   │   └── 用户消息 2a
│   │       └── 助手回复 2a
│   └── 助手回复 1b (分支)
│       └── 用户消息 2b
│           └── 助手回复 2b
└── 用户消息 1 (另一个分支)
    └── 助手回复 1c
```

### 组件架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ConversationTree                         │
│  ┌─────────────────────────────────────────────────────────┤
│  │                 MessageNode                             │
│  │  ┌─────────────────────────────────────────────────────┤
│  │  │               Message                               │
│  │  │  ┌─────────────────────────────────────────────────┤
│  │  │  │             Part (消息内容)                      │
│  │  │  │  • TextPart (文本)                             │
│  │  │  │  • ThoughtPart (思考)                          │
│  │  │  │  • ContentPart (内容)                          │
│  │  │  │  • FilePart (文件)                             │
│  │  │  │  • ImagePart (图片)                            │
│  │  │  └─────────────────────────────────────────────────┘
│  │  └─────────────────────────────────────────────────────┘
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## 核心类

### 1. Part 类族

消息内容的基础构建块：

```python
class Part(ABC):
    """消息内容部分的抽象基类"""
    def __init__(self, content: Optional[str] = None, mime_type: Optional[str] = None):
        self.content = content
        self.mime_type = mime_type

class TextPart(Part):
    """文本内容部分"""
    pass

class ThoughtPart(TextPart):
    """思考过程部分，用于记录模型的内部思考"""
    pass

class ContentPart(TextPart):
    """正式回答内容部分"""
    pass

class FilePart(Part):
    """通用文件部分"""
    pass

class ImagePart(FilePart):
    """图像内容部分"""
    pass
```

### 2. MessageNode 类

对话树中的节点：

```python
class MessageNode:
    def __init__(self, message_obj: 'Message', parent: Optional['MessageNode'] = None):
        self.id = message_obj.id
        self.message = message_obj
        self.parent = parent
        self.children: List['MessageNode'] = []
        self.tree = None
    
    def add_child(self, child_node: 'MessageNode'):
        """添加子节点"""
        pass
    
    def remove_child(self, child_node: 'MessageNode'):
        """移除子节点"""
        pass
    
    def get_all_children(self) -> List['MessageNode']:
        """获取所有子节点（递归）"""
        pass
```

### 3. ConversationTree 类

对话树的主要管理类：

```python
class ConversationTree:
    def __init__(self, message_cls: "Message", history_cls: "ChatHistory", 
                 system_prompt: Optional[str] = None, file_manager: Optional["FileManager"] = None):
        self.id = str(uuid.uuid4())
        self.root: Optional[MessageNode] = None
        self.current_node = self.root
        self.nodes: Dict[str, MessageNode] = {}
    
    def add_message(self, role: str, text: Optional[str] = None, **kwargs) -> MessageNode:
        """添加消息到当前节点"""
        pass
    
    def add_user_message(self, text: str, **kwargs) -> MessageNode:
        """添加用户消息"""
        pass
    
    def add_assistant_message(self, text: str, **kwargs) -> MessageNode:
        """添加助手消息"""
        pass
    
    def switch_to_node(self, node: str | MessageNode) -> MessageNode:
        """切换到指定节点"""
        pass
    
    def get_history(self, max_depth: int = 10000) -> "ChatHistory":
        """获取当前路径的历史记录"""
        pass
    
    def pretty_print(self):
        """美观地打印对话树结构"""
        pass
```

## 主要功能

### 1. 基本对话管理

```python
from llm.history import ConversationTree
from llm.providers import Message, ChatHistory

# 创建对话树
tree = ConversationTree(
    message_cls=Message,
    history_cls=ChatHistory,
    system_prompt="You are a helpful assistant."
)

# 添加用户消息
user_node = tree.add_user_message("Hello, how are you?")

# 添加助手回复
assistant_node = tree.add_assistant_message("I'm doing well, thank you!")

# 获取对话历史
history = tree.get_history()
print(history.simple_print())
```

### 2. 分支对话

```python
# 从某个节点创建分支
tree.switch_to_node(user_node)

# 添加不同的助手回复（创建分支）
alternative_response = tree.add_assistant_message("I'm having a great day!")

# 继续这个分支的对话
tree.add_user_message("That's wonderful to hear!")
```

### 3. 多模态消息

```python
# 添加包含图片的消息
tree.add_user_message(
    text="What do you see in this image?",
    local_images=["path/to/image.jpg"]
)

# 添加包含思考过程的消息
tree.add_assistant_message(
    text="I can see a beautiful landscape.",
    thought="Let me analyze this image carefully..."
)

# 添加包含文件的消息
tree.add_user_message(
    text="Please review this document",
    local_files=["path/to/document.pdf"]
)
```

### 4. 节点操作

```python
# 切换到特定节点
tree.switch_to_node(node_id_or_object)

# 移除节点及其子节点
tree.remove_node_from_tree_with_children(node)

# 移除节点但保留子节点
tree.remove_node_from_tree_with_concrete_children(node)

# 获取从特定节点开始的历史
history = tree.get_history_from_node(node, max_depth=5)
```

### 5. 可视化

```python
# 打印对话树结构
tree.pretty_print()

# 输出示例：
# └── [SYSTEM] (id: abc123) Parts: TextPart: 'You are a helpful assistant.'
#     ├── [USER] (id: def456) Parts: TextPart: 'Hello, how are you?'
#     │   ├── [ASSISTANT] (id: ghi789) Parts: TextPart: 'I'm doing well, thank you!'
#     │   └── [ASSISTANT] (id: jkl012) Parts: TextPart: 'I'm having a great day!'
#     └── [USER] (id: mno345) Parts: TextPart: 'What's the weather like?'
```

## 错误处理

模块提供了完整的错误处理机制：

```python
from llm.history import HistoryError, NodeNotFoundError, ProviderError, UnsupportedModalityError

try:
    tree.switch_to_node("nonexistent_node_id")
except NodeNotFoundError as e:
    print(f"节点未找到: {e}")

try:
    tree.add_user_message(local_images=["image.jpg"])  # 没有配置 file_manager
except ProviderError as e:
    print(f"提供商错误: {e}")
```

## 与 Providers 模块集成

History 模块与 Providers 模块完全兼容：

```python
from llm.providers import get_provider_manager, ChatRequest
from llm.history import ConversationTree

# 创建提供商
manager = get_provider_manager()
provider = await manager.create_provider("openai", api_key="your-key")

# 创建对话树
tree = ConversationTree(
    message_cls=provider.message_cls,
    history_cls=provider.history_cls,
    system_prompt="You are a helpful assistant."
)

# 添加用户消息
tree.add_user_message("Hello!")

# 获取历史并发送给 LLM
history = tree.get_history()
request = ChatRequest(
    messages=[msg.to_chat_message() for msg in history.messages],
    model="gpt-4o"
)

response = await provider.chat(request)

# 添加 LLM 回复到对话树
tree.add_assistant_message(response.content)
```

## 使用场景

1. **对话系统**: 构建支持分支对话的聊天机器人
2. **决策树**: 创建基于用户选择的交互式决策流程
3. **教学系统**: 根据学生回答提供不同的学习路径
4. **游戏对话**: 实现复杂的游戏角色对话系统
5. **调试工具**: 测试不同的 prompt 和回复策略

## 性能考虑

- 对话树会在内存中保存所有节点
- 对于大型对话树，建议定期清理不需要的分支
- 使用 `max_depth` 参数限制历史记录长度
- 文件处理依赖于配置的 `file_manager`

## 开发状态

- ✅ 基础树形结构
- ✅ 节点管理
- ✅ 多模态支持
- ✅ 历史记录生成
- ✅ 错误处理
- ✅ 可视化输出
- ⏳ 持久化存储
- ⏳ 历史记录压缩
- ⏳ 更多文件类型支持 