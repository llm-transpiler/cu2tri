# -*- coding: utf-8 -*-
"""
LLM提供商类型定义 - 三层架构
平台 → 厂商 → 模型的优雅分级设计
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set
import re
try: # 防止循环引用
    from .config import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
except ImportError:
    DEFAULT_TIMEOUT = 300
    DEFAULT_MAX_RETRIES = 5

# ============ 第一层：平台类型 ============

class PlatformCategory(str, Enum):
    """平台类别"""
    AGGREGATOR = "aggregator"       # 聚合平台 - 支持多厂商模型
    OFFICIAL = "official"           # 官方平台 - 厂商自有API
    LOCAL = "local"                 # 本地平台 - 本地推理服务
    CUSTOM = "custom"               # 自定义平台 - 用户自定义

class PlatformType(str, Enum):
    """部署平台/公司, 付费入口"""
    # 聚合平台, 没有自研模型, 转发模型厂商提供的API, 或可能提供其他家的部署
    OPENROUTER = "openrouter"
    # LITELLM = "litellm"
    # TOGETHER = "together"
    # REPLICATE = "replicate"
    # HUGGINGFACE = "huggingface"
    # AZURE_OPENAI = "azure_openai"
    # SILICONFLOW = "siliconflow"
    
    # 提供API的公司, 具有自己的部署
    OPENAI_OFFICIAL = "openai"
    ANTHROPIC_OFFICIAL = "anthropic"
    GOOGLE_OFFICIAL = "google"
    DEEPSEEK_OFFICIAL = "deepseek"
    ZHIPU_OFFICIAL = "zhipu"
    # COHERE_OFFICIAL = "cohere"
    # MISTRAL_OFFICIAL = "mistral"
    
    # # 本地平台
    VLLM = "vllm"
    # OLLAMA = "ollama"
    # TEXT_GENERATION_WEBUI = "text_generation_webui"
    # LLAMACPP = "llamacpp"
    # TRANSFORMERS = "transformers"
    # FASTCHAT = "fastchat"
    # LMDEPLOY = "lmdeploy"
    
    # # 自定义平台
    # CUSTOM_HTTP = "custom_http"
    # CUSTOM_GRPC = "custom_grpc"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

class SDKType(str, Enum):
    """SDK类型, 只考虑python的SDK"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENAI = "genai"
    
    VLLM = "vllm"

# ============ 第二层：厂商类型 ============

class VendorType(str, Enum):
    """模型厂商名称, 用于openrouter中的{vendor_type}/{model_name}及类似场景"""
    # 商业厂商
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CLAUDE = "anthropic"
    GOOGLE = "google"
    # COHERE = "cohere"
    # MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"             # 智谱AI
    # MOONSHOT = "moonshot"     # 月之暗面
    # BAICHUAN = "baichuan"     # 百川智能
    # ALIBABA = "alibaba"       # 阿里巴巴（通义千问）
    # BAIDU = "baidu"           # 百度（文心一言）
    # TENCENT = "tencent"       # 腾讯（混元）
    
    # # 开源厂商/组织
    # META = "meta"                 # Meta (Llama系列)
    # MICROSOFT = "microsoft"       # Microsoft (Phi系列)
    # NVIDIA = "nvidia"             # NVIDIA (Nemotron系列)
    # DATABRICKS = "databricks"     # Databricks (DBRX)
    # APPLE = "apple"               # Apple (OpenELM)
    # IBM = "ibm"                   # IBM (Granite)
    # SALESFORCE = "salesforce"     # Salesforce (CodeGen)
    # BIGSCIENCE = "bigscience"     # BigScience (BLOOM)
    # HUGGINGFACE = "huggingface"   # HuggingFace自有模型
    
    # # 开源社区
    # ALPACA = "alpaca"         # Alpaca系列
    # VICUNA = "vicuna"         # Vicuna系列
    # WIZARD = "wizard"         # WizardLM系列
    # ORCA = "orca"             # Orca系列
    # OPEN_CHAT = "open_chat"   # OpenChat系列
    
    # 聚合API自行部署
    OPENROUTER_DEEPSEEK = "deepseek"
    
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

# ============ 第三层：模型定义 ============

@dataclass
class ModelSpec:
    """模型规格定义"""
    name: str                                       # 模型名称
    vendor: VendorType                              # 所属厂商
    size: Optional[str] = None                      # 模型大小 (7B, 13B, 70B等)
    variant: Optional[str] = None                   # 变体 (chat, instruct, base等)
    context_length: Optional[int] = None            # 上下文长度
    input_token_limit: Optional[int] = None         # 输入token限制
    output_token_limit: Optional[int] = None        # 输出token限制
    is_thinking: bool = False                       # 是否支持思考
    supports_streaming_output: bool = False         # 是否支持流式输出
    supports_function_calling_input: bool = False   # 是否支持函数调用
    supports_picture_input: bool = False            # 是否支持图片
    supports_audio_input: bool = False              # 是否支持音频
    supports_video_input: bool = False              # 是否支持视频
    supports_pdf_input: bool = False                # 是否支持pdf
    supports_structured_output: bool = False        # 是否支持结构化输出
    supports_code: bool = False                     # 是否支持代码
    is_open_source: bool = False                    # 是否开源
    # https://ai.google.dev/gemini-api/docs/pricing?hl=zh-cn
    input_cost_per_million_tokens: Optional[float] = None # 输入价格/百万token
    output_cost_per_million_tokens: Optional[float] = None # 输出价格/百万token
    aliases: List[str] = field(default_factory=list)  # 别名列表


# ============ 平台支持矩阵 ============

@dataclass 
class PlatformInfo:
    """平台信息"""
    type_: PlatformType
    category: PlatformCategory
    name: str
    description: str
    requires_api_key: bool = True
    # supports_streaming: bool = True
    # supports_multimodal: bool = False
    http_base_url: Optional[str] = None
    base_url: str = ""  # 新增的base_url属性
    supported_vendors: Set[VendorType] = field(default_factory=set)
    supported_models: Set[str] = field(default_factory=set)
    supported_sdks: Set[SDKType] = field(default_factory=set)  # 支持的SDK类型
    model_name_format: str = "{model}"  # 模型名称格式化模板

# 平台注册表
PLATFORM_REGISTRY: Dict[PlatformType, PlatformInfo] = {
    # ============ 聚合平台 ============
    PlatformType.OPENROUTER: PlatformInfo(
        type_=PlatformType.OPENROUTER,
        category=PlatformCategory.AGGREGATOR,
        name="OpenRouter",
        description="OpenRouter aggregation platform supporting multi-vendor models",
        http_base_url="https://openrouter.ai/api/v1",
        base_url="https://openrouter.ai/api/v1",
        supported_vendors={
            VendorType.OPENAI, VendorType.DEEPSEEK, VendorType.OPENROUTER_DEEPSEEK,
            VendorType.GOOGLE,VendorType.ANTHROPIC
        },
        supported_sdks={SDKType.OPENAI},
        model_name_format="{vendor}/{model}"
    ),
    
    # ============ 模型厂商官方部署平台 ============
    PlatformType.OPENAI_OFFICIAL: PlatformInfo(
        type_=PlatformType.OPENAI_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="OpenAI Official",
        description="OpenAI官方API",
        http_base_url="https://api.openai.com/v1",
        base_url="https://api.openai.com/v1",
        supported_vendors={VendorType.OPENAI},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.ANTHROPIC_OFFICIAL: PlatformInfo(
        type_=PlatformType.ANTHROPIC_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Anthropic Official",
        description="Anthropic官方API (Claude)",
        http_base_url="https://api.anthropic.com/v1/", # https://docs.anthropic.com/en/api/overview#curl
        base_url="https://api.anthropic.com/v1/", # https://docs.anthropic.com/en/api/openai-sdk#getting-started-with-the-openai-sdk
        supported_vendors={VendorType.ANTHROPIC},
        supported_sdks={SDKType.ANTHROPIC, SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.GOOGLE_OFFICIAL: PlatformInfo(
        type_=PlatformType.GOOGLE_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Google Official",
        description="Google genai official API",
        http_base_url="https://generativelanguage.googleapis.com/v1beta", # https://ai.google.dev/gemini-api/docs/quickstart#rest
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # https://ai.google.dev/gemini-api/docs/openai?hl=zh-cn
        supported_vendors={VendorType.GOOGLE},
        supported_sdks={SDKType.GENAI, SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.DEEPSEEK_OFFICIAL: PlatformInfo(
        type_=PlatformType.DEEPSEEK_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="DeepSeek Official",
        description="DeepSeek official API",
        http_base_url="https://api.deepseek.com", # https://api-docs.deepseek.com/zh-cn/
        base_url="https://api.deepseek.com", # https://api-docs.deepseek.com/zh-cn/
        supported_vendors={VendorType.DEEPSEEK},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.ZHIPU_OFFICIAL: PlatformInfo(
        type_=PlatformType.ZHIPU_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Zhipu AI Official",
        description="智谱AI官方API",
        http_base_url="https://open.bigmodel.cn/api/paas/v4/",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        supported_vendors={VendorType.ZHIPU},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    # ============ 本地平台 ============
    PlatformType.VLLM: PlatformInfo(
        type_=PlatformType.VLLM,
        category=PlatformCategory.LOCAL,
        name="vLLM",
        description="vLLM local inference server",
        http_base_url="http://localhost:8000/v1",
        base_url="http://localhost:8000/v1",
        supported_vendors=set(),  # vLLM 可以运行任何厂商的模型
        supported_sdks={SDKType.OPENAI, SDKType.VLLM},
        model_name_format="{model}"
    ),
}

# SDKType -> PlatformType
SDK_REGISTRY: Dict[SDKType, List[PlatformType]] = {
    SDKType.OPENAI: [PlatformType.OPENAI_OFFICIAL, PlatformType.ANTHROPIC_OFFICIAL, PlatformType.GOOGLE_OFFICIAL, PlatformType.DEEPSEEK_OFFICIAL, PlatformType.OPENROUTER],
    SDKType.ANTHROPIC: [PlatformType.ANTHROPIC_OFFICIAL],
    SDKType.GENAI: [PlatformType.GOOGLE_OFFICIAL],
    SDKType.VLLM: [PlatformType.VLLM],
}

# ================== Per vendor ==================

# ================== Google Gemini ==================
GEMINI_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="gemini-2.5-pro",
        vendor=VendorType.GOOGLE,
        is_thinking=True,
        input_token_limit=1048576, # 1M
        output_token_limit=65536, # 66K
        supports_streaming_output=True,
        supports_picture_input=True,
        supports_structured_output=True,
        supports_pdf_input=True,
        supports_audio_input=True,
        supports_video_input=True,
        supports_function_calling_input=True,
    ),
    ModelSpec(
        name="gemini-2.5-flash",
        vendor=VendorType.GOOGLE,
        is_thinking=True,
        input_token_limit=1048576, # 1M
        output_token_limit=65536, # 66K
        supports_streaming_output=True,
        supports_picture_input=True,
        supports_structured_output=True,
        supports_pdf_input=True,
        supports_audio_input=True,
        supports_video_input=False,
        supports_function_calling_input=True,
    ),
]

OPENAI_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="gpt-o3-mini",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-o3",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-4.1",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-4o",
        vendor=VendorType.OPENAI,
        supports_picture_input=True,
        aliases=["gpt-4-omni"]
    ),
    ModelSpec(
        name="gpt-4o-mini",
        vendor=VendorType.OPENAI,
        supports_picture_input=True,
        aliases=["gpt-4-omni-mini"]
    ),
]

ANTHROPIC_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="claude-4-sonnet",
        vendor=VendorType.ANTHROPIC,
        is_thinking=True,
        supports_picture_input=True,
    ),
    ModelSpec(
        name="claude-4-opus",
        vendor=VendorType.ANTHROPIC,
        is_thinking=True,
        supports_picture_input=True,
    ),
]

DEEPSEEK_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="deepseek-r1-0528",
        vendor=VendorType.DEEPSEEK,
        is_thinking=True,
    ),
    ModelSpec(
        name="deepseek-v3-0324",
        vendor=VendorType.DEEPSEEK,
    ),
]

# VendorType -> List[ModelSpec]
OPENROUTER_EXCLUSIVE_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="deepseek-r1-0528:free",
        vendor=VendorType.OPENROUTER_DEEPSEEK,
    ),
    ModelSpec(
        name="deepseek-chat-v3-0324:free",
        vendor=VendorType.OPENROUTER_DEEPSEEK,
    ),
]

MODEL_REGISTRY_LIST = [
    OPENAI_MODEL_REGISTRY,
    ANTHROPIC_MODEL_REGISTRY,
    GEMINI_MODEL_REGISTRY,
    DEEPSEEK_MODEL_REGISTRY,
    OPENROUTER_EXCLUSIVE_MODEL_REGISTRY,
]
def get_vendor_model_registry() -> Dict[VendorType, List[ModelSpec]]:
    """
    获取厂商模型注册表
    return {
        VendorType.OPENAI: OPENAI_MODEL_REGISTRY,
        VendorType.ANTHROPIC: ANTHROPIC_MODEL_REGISTRY,
        VendorType.GOOGLE: GEMINI_MODEL_REGISTRY,
        VendorType.DEEPSEEK: DEEPSEEK_MODEL_REGISTRY,
        VendorType.OPENROUTER_DEEPSEEK: OPENROUTER_EXCLUSIVE_MODEL_REGISTRY,
    }
    """
    vendor_registry = {}
    for registry in MODEL_REGISTRY_LIST:
        for model_spec in registry:
            vendor = model_spec.vendor
            if vendor not in vendor_registry:
                vendor_registry[vendor] = []
            vendor_registry[vendor].append(model_spec)
    return vendor_registry

# VendorType -> List[ModelSpec]
VENDOR_MODEL_REGISTRY = get_vendor_model_registry()

# ============ 工具函数 ============

def get_platform_info(platform_type: Union[str, PlatformType]) -> Optional[PlatformInfo]:
    """获取平台信息"""
    if platform_type is None:
        return None
    if isinstance(platform_type, str):
        platform_type = PlatformType(platform_type)  # 让ValueError向上传播
    return PLATFORM_REGISTRY.get(platform_type)

def get_model_spec(model_name: str, vendor: Optional[VendorType] = None) -> Optional[ModelSpec]:
    """获取模型规格"""
    if not model_name:
        return None
    
    # 如果指定了厂商，直接在该厂商下查找
    if vendor and vendor in VENDOR_MODEL_REGISTRY:
        for spec in VENDOR_MODEL_REGISTRY[vendor]:
            if spec.name == model_name or model_name in spec.aliases:
                return spec
    
    # 在所有厂商中查找
    for vendor_models in VENDOR_MODEL_REGISTRY.values():
        for spec in vendor_models:
            if spec.name == model_name or model_name in spec.aliases:
                return spec
    
    return None

def get_supported_models(platform_type: PlatformType, vendor: Optional[VendorType] = None) -> List[str]:
    """获取平台支持的模型列表"""
    if platform_type is None:
        return []
        
    platform_info = get_platform_info(platform_type)
    if not platform_info:
        return []
    
    models = []
    for vendor_type, vendor_models in VENDOR_MODEL_REGISTRY.items():
        # 检查厂商是否支持
        if vendor_type not in platform_info.supported_vendors:
            continue
        
        # 如果指定了厂商，检查是否匹配
        if vendor and vendor_type != vendor:
            continue
        
        for spec in vendor_models:
            # 格式化模型名称
            formatted_name = platform_info.model_name_format.format(
                vendor=str(spec.vendor),
                model=spec.name
            )
            models.append(formatted_name)
    
    return sorted(models)

def get_platforms_by_category(category: PlatformCategory) -> List[PlatformType]:
    """按类别获取平台类型列表"""
    return [info.type_ for info in PLATFORM_REGISTRY.values() if info.category == category]

def get_vendors_by_platform(platform_type: PlatformType) -> Set[VendorType]:
    """获取平台支持的厂商列表"""
    platform_info = get_platform_info(platform_type)
    return platform_info.supported_vendors if platform_info else set()
