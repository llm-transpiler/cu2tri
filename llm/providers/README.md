# LLM Providers 模块

一个统一的大语言模型提供商管理系统，支持多种平台和厂商的模型调用。

## 🏗️ 架构设计

本模块采用简化的三层架构设计：

```
平台层 (Platform) → 厂商层 (Vendor) → 模型层 (Model)
```

### 核心组件

```
llm/providers/
├── types.py          # 类型定义和注册表
├── config.py         # 配置管理系统
├── base.py           # 基础接口和统一数据模型
├── factory.py        # 工厂和管理器
├── impl.py           # 具体提供商实现
└── __init__.py       # 导出接口
```

## 📋 支持的平台

### 聚合平台
- **OpenRouter** - 支持多厂商模型的聚合平台

### 官方API
- **OpenAI** - GPT系列模型
- **Anthropic** - Claude系列模型  
- **Google** - Gemini系列模型
- **DeepSeek** - DeepSeek系列模型
- **ZhipuAI** - 智谱AI系列模型

### 本地部署
- **vLLM** - 本地推理服务

## 🚀 快速开始

### 基本使用

```python
from llm.providers import (
    PlatformConfig, PlatformType, 
    ProviderFactory, get_config_manager
)

# 1. 创建配置
config = PlatformConfig(
    platform_type=PlatformType.OPENAI_OFFICIAL,
    api_key="your-api-key",
    preferred_models=["gpt-4o", "gpt-4o-mini"],
    temperature=0.7,
    max_tokens=4096
)

# 2. 创建提供商
provider = ProviderFactory.create_provider(config)

# 3. 初始化并使用
await provider.initialize()

# 4. 发送请求
from llm.providers import ChatMessage, ChatRequest
request = ChatRequest(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="gpt-4o-mini"
)

response = await provider.chat(request)
print(response.content)

# 5. 流式请求
async for chunk in provider.stream_chat(request):
    print(chunk.content, end="")
```

### 使用配置管理器

```python
from llm.providers import get_config_manager, get_provider_manager

# 配置管理
config_manager = get_config_manager()
config_manager.set_config("openai", PlatformConfig(
    platform_type="openai",
    api_key="your-key"
))
config_manager.save_config()

# 提供商管理
provider_manager = get_provider_manager()
provider_manager.load_from_config_manager()

# 初始化并使用
await provider_manager.initialize_provider("openai")
provider = provider_manager.get_initialized_provider("openai")
```

## 🔧 配置系统

### PlatformConfig 配置项

```python
@dataclass
class PlatformConfig:
    platform_type: Union[str, PlatformType]  # 平台类型
    enabled: bool = True                     # 是否启用
    
    # 认证配置
    api_key: Optional[str] = None            # API密钥
    api_base: Optional[str] = None           # 自定义API地址
    organization: Optional[str] = None        # 组织ID
    
    # 请求配置
    timeout: int = 300                       # 超时时间
    max_retries: int = 5                     # 重试次数
    
    # 模型配置
    preferred_models: List[str] = []         # 首选模型
    model_aliases: Dict[str, str] = {}       # 模型别名
    
    # 默认参数
    temperature: Optional[float] = None      # 温度
    max_tokens: Optional[int] = None         # 最大tokens
    top_p: Optional[float] = None            # top_p参数
    
    # 特性配置
    supports_streaming: bool = True          # 流式输出
    supports_function_calling: bool = False # 函数调用
    supports_vision: bool = False            # 视觉能力
```

### 配置文件格式

```yaml
# llm_providers_config.yaml
platforms:
  openai:
    platform_type: openai
    api_key: "your-openai-key"
    preferred_models: ["gpt-4o", "gpt-4o-mini"]
    temperature: 0.7
    max_tokens: 4096
  
  openrouter:
    platform_type: openrouter
    api_key: "your-openrouter-key" 
    preferred_models: ["openai/gpt-4o", "deepseek/deepseek-r1-0528:free"]

default_platform: "openai"
default_model: "gpt-4o-mini"
fallback_models: ["gpt-4o-mini", "deepseek/deepseek-r1-0528:free"]
```

## 🎯 核心接口

### 统一数据模型

```python
# 聊天消息
@dataclass
class ChatMessage:
    role: str          # user, assistant, system
    content: str       # 消息内容
    metadata: Dict     # 元数据
    timestamp: str     # 时间戳

# 聊天请求
@dataclass
class ChatRequest:
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # ... 其他参数

# 聊天响应
@dataclass
class ChatResponse:
    content: str                    # 响应内容
    model: str                      # 使用的模型
    usage: Optional[Dict] = None    # 使用统计
    finish_reason: Optional[str] = None
    processing_time: Optional[float] = None

# 流式响应块
@dataclass
class StreamChunk:
    content: str                    # 块内容
    finish_reason: Optional[str] = None
```

### Provider 基类

```python
class Provider(ABC):
    """统一的LLM提供商基类"""
    
    # 生命周期管理
    async def initialize() -> None
    async def close() -> None
    
    # 核心接口
    @abstractmethod
    async def chat(request: ChatRequest) -> ChatResponse
    
    @abstractmethod  
    async def stream_chat(request: ChatRequest) -> AsyncGenerator[StreamChunk, None]
    
    # 可选接口
    async def list_models() -> List[str]
    async def health_check() -> bool
    
    # 验证
    def validate_request(request: ChatRequest) -> List[str]
```

## 🏭 工厂模式

### ProviderFactory

```python
# 注册提供商
ProviderFactory.register_provider(PlatformType.OPENAI_OFFICIAL, OpenAIProvider)

# 创建提供商
provider = ProviderFactory.create_provider(config)

# 从配置名创建
provider = ProviderFactory.create_from_config_name("openai")
```

### ProviderManager

```python  
manager = ProviderManager()

# 添加提供商
manager.add_provider("openai", config)

# 初始化
await manager.initialize_provider("openai")

# 获取提供商
provider = manager.get_initialized_provider("openai")

# 生命周期管理
async with manager:
    # 使用提供商
    pass  # 自动关闭
```

## 🔌 扩展新平台

### 1. 定义平台信息

```python
# types.py
class PlatformType(str, Enum):
    NEW_PLATFORM = "new_platform"

PLATFORM_REGISTRY[PlatformType.NEW_PLATFORM] = PlatformInfo(
    type_=PlatformType.NEW_PLATFORM,
    category=PlatformCategory.OFFICIAL,
    name="New Platform",
    description="New platform description",
    base_url="https://api.newplatform.com/v1",
    supported_vendors={VendorType.NEW_VENDOR},
    supported_sdks={SDKType.OPENAI}
)
```

### 2. 实现Provider

```python
# impl.py
class NewPlatformProvider(OpenAICompatibleProvider):
    """新平台提供商"""
    
    async def _initialize_client(self) -> None:
        # 自定义初始化逻辑
        await super()._initialize_client()
```

### 3. 注册提供商

```python
# impl.py
ProviderFactory.register_provider(PlatformType.NEW_PLATFORM, NewPlatformProvider)
```

## 🛠️ 异常处理

```python
from llm.providers import (
    ProviderError, AuthenticationError, RateLimitError,
    ModelNotFoundError, ValidationError, NetworkError
)

try:
    response = await provider.chat(request)
except AuthenticationError as e:
    print(f"认证失败: {e}")
except RateLimitError as e:
    print(f"限流: {e}")
except ModelNotFoundError as e:
    print(f"模型不存在: {e}")
except ProviderError as e:
    print(f"提供商错误: {e}")
```

## 📊 类型系统

### 平台类型层次

```python
PlatformCategory:
├── AGGREGATOR  # 聚合平台
├── OFFICIAL    # 官方API  
├── LOCAL       # 本地部署
└── CUSTOM      # 自定义

PlatformType:
├── OPENROUTER         # 聚合
├── OPENAI_OFFICIAL    # 官方
├── ANTHROPIC_OFFICIAL
├── GOOGLE_OFFICIAL
├── DEEPSEEK_OFFICIAL
├── ZHIPU_OFFICIAL
└── VLLM              # 本地
```

### 模型规格

```python
@dataclass
class ModelSpec:
    name: str                               # 模型名称
    vendor: VendorType                      # 厂商
    context_length: Optional[int] = None    # 上下文长度
    supports_streaming_output: bool = False # 流式输出
    supports_function_calling_input: bool = False # 函数调用
    supports_picture_input: bool = False    # 图片输入
    is_thinking: bool = False               # 思考模式
    # ... 更多属性
```

## 🧪 测试

```bash
# 运行所有测试
python -m pytest llm/unittests/

# 运行特定测试
python -m pytest llm/unittests/test_providers_config.py
python -m pytest llm/unittests/test_providers_factory.py

# 集成测试
python llm/unittests/run_tests.py
```

## 📝 最佳实践

### 1. 配置管理
- 使用环境变量存储敏感信息
- 配置文件与代码分离
- 支持多环境配置

### 2. 错误处理
- 统一异常类型
- 详细错误信息
- 优雅降级

### 3. 资源管理
- 及时释放连接
- 使用异步上下文管理器
- 合理设置超时

### 4. 性能优化
- 连接复用
- 合理的重试策略
- 并发控制

## 🔄 版本兼容性

本模块保持向后兼容，支持旧版本的历史接口：

```python
# 兼容性接口仍然可用
from llm.providers import Message, ChatHistory, Part
```

## 📈 未来计划

- [ ] 支持更多LLM平台
- [ ] 函数调用功能增强
- [ ] 多模态能力扩展
- [ ] 性能监控和指标
- [ ] 缓存和批处理优化

---

**版本**: 1.0.0  
**维护者**: LLM Providers Team 