# -*- coding: utf-8 -*-
"""
内核开发服务数据模型
定义编译、测试、性能测试相关的请求、响应和配置模型
"""
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Union
from enum import Enum
import time

class KernelType(str, Enum):
    """内核类型枚举"""
    CUDA = "cuda"
    TRITON = "triton"
    PYTORCH = "pytorch"

class GPUType(str, Enum):
    """GPU类型枚举"""
    L20 = "l20"
    H100 = "h100"
    A100 = "a100"
    AMDGPU = "amdgpu"
    CAMBRICON_NPU = "cambricon_npu"

class TestStage(str, Enum):
    """测试阶段枚举"""
    DEVELOPMENT = "development"  # 开发阶段：使用GPU 0,1的L20
    PRODUCTION = "production"    # 生产阶段：使用GPU 2-5的H100

class TaskType(str, Enum):
    """任务类型枚举"""
    COMPILE = "compile"
    FUNCTIONAL_TEST = "functional_test"
    PERFORMANCE_TEST = "performance_test"
    LLM_GENERATION = "llm_generation"
    SMT_VERIFICATION = "smt_verification"
    RAG_QUERY = "rag_query"
    WEB_SEARCH = "web_search"

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class GPUInfo:
    """GPU信息"""
    gpu_id: int
    gpu_type: GPUType
    memory_total: int  # MB
    memory_used: int   # MB
    utilization: float  # 0-100%
    temperature: float  # 摄氏度
    is_available: bool
    current_task_id: Optional[str] = None

@dataclass
class KernelCode:
    """内核代码"""
    kernel_type: KernelType
    code: str
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    compile_flags: List[str] = field(default_factory=list)

@dataclass
class TestInput:
    """测试输入数据"""
    input_shapes: List[tuple]
    input_dtypes: List[str]
    test_data: Optional[Dict[str, Any]] = None
    custom_inputs: Optional[List[Any]] = None

@dataclass
class CompileRequest:
    """编译请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    kernel_code: KernelCode
    build_dir: Optional[str] = None
    verbose: bool = False
    timestamp: float = field(default_factory=time.time)

@dataclass
class TestRequest:
    """功能测试请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    kernel_code: KernelCode
    reference_code: Optional[KernelCode] = None  # 参考实现
    test_inputs: TestInput
    tolerance: Dict[str, float] = field(default_factory=lambda: {"atol": 1e-1, "rtol": 1e-1})
    gpu_requirements: List[int] = field(default_factory=list)  # 指定GPU ID
    stage: TestStage = TestStage.DEVELOPMENT
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerfRequest:
    """性能测试请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    kernel_code: KernelCode
    reference_codes: List[KernelCode] = field(default_factory=list)  # 多个参考实现
    test_inputs: TestInput
    benchmark_config: Dict[str, Any] = field(default_factory=dict)
    gpu_requirements: List[int] = field(default_factory=list)
    stage: TestStage = TestStage.DEVELOPMENT
    wait_for_idle: bool = True  # 是否等待GPU空闲
    timestamp: float = field(default_factory=time.time)

@dataclass
class LLMRequest:
    """LLM交互请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    task_type: Literal["generate", "optimize", "debug", "explain"]
    context: Dict[str, Any]  # 包含当前代码、错误信息、性能数据等
    user_message: str
    cuda_code: Optional[str] = None
    triton_code: Optional[str] = None
    error_info: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None
    model_name: str = "deepseek/deepseek-r1:free"
    max_tokens: int = 4096
    temperature: float = 0.7
    timestamp: float = field(default_factory=time.time)

@dataclass
class RAGRequest:
    """RAG查询请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    query: str
    knowledge_base: str = "triton_docs"  # 知识库名称
    top_k: int = 5
    similarity_threshold: float = 0.7
    timestamp: float = field(default_factory=time.time)

@dataclass
class WebSearchRequest:
    """网络搜索请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    query: str
    search_type: Literal["triton_docs", "cuda_docs", "general"] = "triton_docs"
    max_results: int = 10
    timestamp: float = field(default_factory=time.time)

@dataclass
class SMTRequest:
    """SMT形式化验证请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    triton_code: str
    cuda_code: str
    verification_properties: List[str] = field(default_factory=list)
    timeout: int = 300  # 验证超时时间（秒）
    timestamp: float = field(default_factory=time.time)

# 统一的请求类型
KernelRequest = Union[
    CompileRequest, TestRequest, PerfRequest, 
    LLMRequest, RAGRequest, WebSearchRequest, SMTRequest
]

@dataclass
class CompileResult:
    """编译结果"""
    success: bool
    build_dir: str
    compile_time: float
    binary_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class TestResult:
    """测试结果"""
    success: bool
    execution_time: float
    correctness_passed: bool
    max_difference: float = 0.0
    gpu_used: List[int] = field(default_factory=list)
    memory_usage: Dict[str, int] = field(default_factory=dict)  # GPU ID -> memory MB
    errors: List[str] = field(default_factory=list)

@dataclass
class PerfResult:
    """性能测试结果"""
    success: bool
    kernel_performance: Dict[str, float]  # kernel_name -> time_ms
    speedup_ratios: Dict[str, float]  # "kernel_vs_reference" -> speedup
    gpu_utilization: Dict[int, float]  # GPU ID -> utilization %
    memory_bandwidth: Dict[int, float]  # GPU ID -> GB/s
    flops: Optional[float] = None  # GFLOPS
    gpu_used: List[int] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResult:
    """LLM交互结果"""
    success: bool
    generated_code: Optional[str] = None
    explanation: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    model_used: str = ""
    tokens_used: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0

@dataclass
class RAGResult:
    """RAG查询结果"""
    success: bool
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    confidence_score: float = 0.0
    knowledge_base_used: str = ""

@dataclass
class WebSearchResult:
    """网络搜索结果"""
    success: bool
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    search_time: float = 0.0

@dataclass
class SMTResult:
    """SMT验证结果"""
    success: bool
    verification_passed: bool
    proof_details: Dict[str, Any] = field(default_factory=dict)
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    verification_time: float = 0.0

@dataclass
class KernelResponse:
    """统一的内核响应"""
    request_id: str
    conversation_id: str
    task_type: TaskType
    status: TaskStatus
    result: Union[CompileResult, TestResult, PerfResult, LLMResult, RAGResult, WebSearchResult, SMTResult]
    processing_time: float
    gpu_info: Optional[List[GPUInfo]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorResponse:
    """错误响应"""
    request_id: str
    conversation_id: str
    task_type: TaskType
    error: str
    error_type: str
    traceback: Optional[str] = None
    gpu_info: Optional[List[GPUInfo]] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationContext:
    """对话上下文"""
    conversation_id: str
    current_cuda_code: Optional[str] = None
    current_triton_code: Optional[str] = None
    compilation_history: List[CompileResult] = field(default_factory=list)
    test_history: List[TestResult] = field(default_factory=list)
    performance_history: List[PerfResult] = field(default_factory=list)
    optimization_iterations: int = 0
    last_error: Optional[str] = None
    context_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GPUResourceConfig:
    """GPU资源配置"""
    development_gpus: List[int] = field(default_factory=lambda: [0, 1])  # L20 GPU IDs
    production_gpus: List[int] = field(default_factory=lambda: [2, 3, 4, 5])  # H100 GPU IDs
    max_concurrent_tasks: int = 4
    memory_threshold: float = 0.8  # 80% 内存使用率阈值
    utilization_threshold: float = 0.5  # 50% 利用率阈值
    temperature_threshold: float = 85.0  # 85°C 温度阈值
    idle_timeout: int = 30  # 空闲超时时间（秒） 