import logging
import os
import shutil
import sys
import json
import subprocess
import argparse
import time
import re
import httpx
import asyncio
from pathlib import Path
from datetime import datetime
from case_config import XPILER_ALL_CASES, LEETCUDA_DYNAMIC_ALL_CASES, LEETCUDA_DYNAMIC_CASES_2
from openai import OpenAI, AsyncOpenAI
from cu2til.prompt.cuda2triton import simple_initial_prompt, feedback_prompt
import dotenv
from llm.client.openai_compat import get_api_param_openai_default
from llm.client.openai_compat import openai_llm_call, async_openai_llm_call, CallingIdentifier, get_api_params_method, make_openai_message_system, make_openai_message_user, make_openai_message_assistant

# Import NVGPU client
sys.path.insert(0, '/workspace/server/nvgpu')
try:
    from server.nvgpu.client import NVGPUClient
    NVGPU_AVAILABLE = True
except ImportError:
    NVGPU_AVAILABLE = False
    print("Warning: NVGPU client not available, will use local execution")
# 执行自动化测试和修复
dotenv.load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CUDA to Triton translation with iterative fixing')
    parser.add_argument('--model', choices=['qwen', 'gpt', 'gemini', 'deepseek', 'claude', 'glm', 'google', 'openrouter_auto'], 
                       default='gpt', help='Model to use for translation')
    parser.add_argument('--console', action='store_true', default=True,
                       help='Output logs to console (default: True)')
    parser.add_argument('--no-console', action='store_true', default=False,
                       help='Disable console output')
    parser.add_argument('--case-types', nargs='*', 
                       help='Specific case types to test (default: all)')
    parser.add_argument('--first-only', action='store_true', default=False,
                       help='Test only the first case from each case type')
    parser.add_argument('--no-perf', action='store_true', default=True,
                       help='Skip performance testing (add --no-perf to check_triton.py)')
    parser.add_argument('--testset', choices=['xpiler', 'leetcuda_dynamic', 'leetcuda_dynamic_2', 'hard'], 
                       default='xpiler', help='Test set to use (default: xpiler)')
    parser.add_argument('--retry-wait', type=int, default=60,
                       help='Wait time in seconds when encountering API overload errors (default: 60)')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum number of retries for API overload errors (default: 5)')
    # NVGPU server options
    parser.add_argument('--use-nvgpu', action='store_true', default=True,
                       help='Use NVGPU server for task execution (default: True)')
    parser.add_argument('--no-nvgpu', action='store_true', default=False,
                       help='Disable NVGPU server and use local execution')
    parser.add_argument('--nvgpu-server', default='http://localhost:8080',
                       help='NVGPU server URL (default: http://localhost:8080)')
    parser.add_argument('--nvgpu-gpu', type=int, default=None,
                       help='Specific GPU ID to use (default: auto-assign)')
    parser.add_argument('--nvgpu-task-type', choices=['functional', 'performance'],
                       default='functional', help='Task type for NVGPU (default: functional)')
    # Async concurrency options
    parser.add_argument('--concurrency', type=int, default=10,
                       help='Maximum number of concurrent tasks (default: 10)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Handle nvgpu flag
if args.no_nvgpu:
    args.use_nvgpu = False

run_model = args.model
get_api_param = get_api_param_openai_default
model_name = "DEFAULT_MODEL_NAME"

# Create both sync and async clients
client = None
async_client = None

if run_model == "qwen":
    # model_name = "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"
    # model_name = "qwen3-coder-480b-a35b-instruct"
    # model_name = "qwen/qwen3-max"
    model_name = "qwen3-vl-235b-a22b-thinking"
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    # )
    # async_client = AsyncOpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    # )
    # client = OpenAI(
    #     base_url='http://127.0.0.1:8003/v1',  # api_base
    #     api_key="EMPTY"
    # )
    # async_client = AsyncOpenAI(
    #     base_url='http://127.0.0.1:8003/v1',
    #     api_key="EMPTY"
    # )
    client = OpenAI(
        base_url="https://cloud.infini-ai.com/maas/v1",
        api_key=os.getenv("INFINI_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://cloud.infini-ai.com/maas/v1",
        api_key=os.getenv("INFINI_API_KEY"),
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OFFICIAL)
elif run_model == "glm":
    model_name = "glm-4.6"
    client = OpenAI(
        base_url="https://cloud.infini-ai.com/maas/v1",
        api_key=os.getenv("INFINI_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://cloud.infini-ai.com/maas/v1",
        api_key=os.getenv("INFINI_API_KEY"),
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OFFICIAL)
elif run_model == "gpt":
    model_name = "openai/gpt-oss-120b"
    # model_name = "openai/gpt-4o"
    # model_name = "openai/gpt-5-codex"
    model_name = "openai/gpt-5-mini"
    # client = OpenAI(
    #     # base_url='http://10.156.112.253:8000/v1',  # 5880x4
    #     # base_url="http://10.208.130.44:8000/v1", # sigma44:a800x8
    #     base_url="http://127.0.0.1:8002/v1", # docker-h20 8001, 6,7
    #     api_key="EMPTY"
    # )
    # async_client = AsyncOpenAI(
    #     base_url="http://127.0.0.1:8002/v1",
    #     api_key="EMPTY"
    # )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OPENROUTER)
elif run_model == "google":
    model_name = "google/gemma-3-27b-it"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
elif run_model == "gemini":
    model_name = "gemini-2.5-pro"
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    async_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    get_api_param = get_api_params_method(CallingIdentifier.GEMINI_OPENAI)
elif run_model == "deepseek":
    model_name = "deepseek-reasoner"
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    async_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    get_api_param = get_api_params_method(CallingIdentifier.DEEPSEEK_OPENAI)
elif run_model == "claude":
    model_name = "anthropic/claude-sonnet-4.5"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    get_api_param = get_api_params_method(
        CallingIdentifier.ANTHROPIC_OPENROUTER)
elif run_model == "openrouter_auto":
    model_name = "openrouter/auto"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OPENROUTER)
else:
    raise ValueError(f"Unsupported model: {run_model}")

def check_network_connectivity(logger):
    """Check network connectivity using curl-like request."""
    test_urls = [
        "https://openrouter.ai",
        "https://www.google.com",
        "https://www.baidu.com"
    ]
    
    for url in test_urls:
        try:
            logger.info(f"🌐 Checking network connectivity to {url}...")
            result = subprocess.run(
                ["curl", "-sL", "-w", "%{http_code}", "-o", "/dev/null", "--max-time", "10", url],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0 and result.stdout.strip() in ["200", "301", "302"]:
                logger.info(f"✅ Network is reachable ({url} returned {result.stdout.strip()})")
                return True
            else:
                logger.warning(f"⚠️  {url} returned code: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to check {url}: {e}")
            continue
    
    logger.error("❌ Network connectivity check failed for all test URLs")
    return False

def is_network_error(error_message):
    """Check if the error is a network-related error."""
    error_str = str(error_message).lower()
    
    network_error_patterns = [
        r'connection error',
        r'connection reset',
        r'peer closed',
        r'timeout',
        r'timed out',
        r'eof occurred',
        r'incomplete chunked read',
        r'broken pipe',
        r'connection refused',
        r'connection aborted',
        r'network is unreachable',
        r'no route to host',
        r'connection pool',
        r'read timed out',
        r'connect timeout',
    ]
    
    for pattern in network_error_patterns:
        if re.search(pattern, error_str):
            return True
    
    return False

def is_retryable_error(error_message):
    """Check if the error is a retryable API overload/rate limit error."""
    error_str = str(error_message).lower()
    
    # Common overload/rate limit patterns
    retryable_patterns = [
        r'overloaded',
        r'rate limit',
        r'quota exceeded',
        r'too many requests',
        r'service unavailable',
        r'temporarily unavailable',
        r'try again later',
        r'error code: 503',
        r'error code: 429',
        r'error code: 502',
        r'status: unavailable',
        r'status: resource_exhausted',
        r'capacity',
        r'busy',
        r'throttled'
    ]
    
    for pattern in retryable_patterns:
        if re.search(pattern, error_str):
            return True
    
    # Network errors are also retryable
    return is_network_error(error_message)

# Constants
DIR_CUDA_ = Path("cuda_")
DIR_TORCH_ = Path("torch_")
DIR_TRITON_ = Path("triton_")
TESTSET_ROOT_DIR = Path(f"/workspace/cu2til/cases/{args.testset}")
TEMPERATURE = 0.35
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MAX_ROUNDS = 5
CONSOLE_OUTPUT = args.console and not args.no_console  # Whether to output to console
CHECK_SUFFIX = "_dynamic"
if "xpiler" in args.testset:
    CHECK_SUFFIX = ""

if args.testset == "leetcuda_dynamic":
    ALL_CASES = LEETCUDA_DYNAMIC_ALL_CASES
elif args.testset == "leetcuda_dynamic_2":
    ALL_CASES = LEETCUDA_DYNAMIC_CASES_2
elif args.testset == "xpiler":
    ALL_CASES = XPILER_ALL_CASES
elif args.testset == "hard":
    ALL_CASES = {"fa": ["fa_cute"]}

# ALL_CASES = {"add": ["add_f16x8_pack"]}
# Setup work directory
model_name_clean = "".join(c if c.isalnum() else "_" for c in model_name.split("/")[-1].lower())
WORK_DIR = Path(__file__).parent.absolute() / f"{model_name_clean}_{args.testset}" / TIMESTAMP
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = WORK_DIR / f"{model_name_clean}_{args.testset}.log"
logger = logging.getLogger('llm_trans')
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Console handler (optional)
handlers = [file_handler]
if CONSOLE_OUTPUT:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)

# Formatter with fixed-width levelname for better alignment
formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(message)s')
for handler in handlers:
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"Using {run_model} model: {model_name}")


def get_first_case_from_each_type():
    """Extract the first case from each case type in ALL_CASES."""
    first_cases = {}
    for case_type, cases in ALL_CASES.items():
        if cases:  # Make sure the list is not empty
            first_cases[case_type] = cases[0]
    return first_cases


def get_failed_cases():
    """Extract failed case types for focused testing."""
    failed_case_types = [
        # "deformable",
        # "depthwiseconv",
        # "gelu",
        # "gemm",
        # "gemv",
        # "layernorm",
        # "maxpool",
        # "mha",
        # "minpool",
        # "relu",
        # "rmsnorm",
        # "sigmoid",
        # "sign",
        # "softmax",
        # "sumpool"
    ]

    failed_cases = {}
    for case_type in failed_case_types:
        if case_type in ALL_CASES:
            failed_cases[case_type] = ALL_CASES[case_type]
        else:
            logger.warning(
                f"Case type '{case_type}' not found in ALL_CASES")

    return failed_cases


# Configure available cases based on command line arguments
def get_available_cases():
    """Get available cases based on command line arguments."""
    if args.first_only:
        cases = get_first_case_from_each_type()
    else:
        cases = ALL_CASES.copy()
    
    # Filter by specified case types if provided
    if args.case_types:
        filtered_cases = {}
        for case_type in args.case_types:
            if case_type in cases:
                filtered_cases[case_type] = cases[case_type]
            else:
                logger.warning(f"Case type '{case_type}' not found in available cases")
        cases = filtered_cases
    
    return cases

AVAILABLE_CASES = get_available_cases()

# Global JSONL log file for real-time statistics
JSONL_LOG_FILE = WORK_DIR / f"{model_name_clean}_{args.testset}.jsonl"
JSONL_LOCK = asyncio.Lock()


async def write_jsonl_log(log_entry):
    """Write a log entry to JSONL file in real-time with async lock."""
    async with JSONL_LOCK:
        try:
            with open(JSONL_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")


def extract_thinking_content(full_response):
    """
    Extract thinking content from response if available.
    Supports various thinking formats: <thought>, <thinking>, etc.
    """
    thinking_patterns = [
        r'<thought>(.*?)</thought>',
        r'<thinking>(.*?)</thinking>',
        r'<think>(.*?)</think>'
    ]
    
    for pattern in thinking_patterns:
        match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


async def save_llm_conversation(test_work_dir, pass_id, round_id, attempt_id, messages, full_response, model_used=None):
    """
    Save LLM conversation to JSONL format with structured metadata.
    Supports pass@n, multiple rounds, and retries.
    
    File structure: logs/conversations/all_conversations.jsonl
    
    Format: Each interaction as a JSONL line:
    {
        "pass": 1,           # Pass number (for pass@n support)
        "round": 1,          # Round number (1=initial, 2+=feedback rounds)
        "attempt": 0,        # Retry attempt (0=initial, 1+=retries)
        "timestamp": "...",  # ISO timestamp
        "interaction_type": "request/response",
        "role": "user/assistant/system",
        "content": "...",    # Message content
        "thinking": "...",   # Optional: thinking content for assistant
        "model_used": "...", # Optional: actual model used (for openrouter auto)
        "success": true      # Whether this attempt was successful
    }
    """
    conversations_dir = test_work_dir / "logs" / "conversations"
    conversations_dir.mkdir(parents=True, exist_ok=True)
    
    # Single JSONL file per case for all conversations
    conversation_file = conversations_dir / "all_conversations.jsonl"
    
    try:
        timestamp = datetime.now().isoformat()
        
        with open(conversation_file, 'a', encoding='utf-8') as f:
            # Write request messages (user/system prompts)
            for msg in messages:
                entry = {
                    "pass": pass_id,
                    "round": round_id,
                    "attempt": attempt_id,
                    "timestamp": timestamp,
                    "interaction_type": "request",
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", "")
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Extract thinking content if available
            thinking_content = extract_thinking_content(full_response)
            
            # Write response (assistant message)
            response_entry = {
                "pass": pass_id,
                "round": round_id,
                "attempt": attempt_id,
                "timestamp": timestamp,
                "interaction_type": "response",
                "role": "assistant",
                "content": full_response,
                "success": True  # Assume success if we got here
            }
            
            # Add model_used if available
            if model_used:
                response_entry["model_used"] = model_used
            
            # Add thinking if available
            if thinking_content:
                response_entry["thinking"] = thinking_content
                # Also save content without thinking for easier use
                response_entry["content_without_thinking"] = re.sub(
                    r'<thought>.*?</thought>|<thinking>.*?</thinking>|<think>.*?</think>',
                    '', full_response, flags=re.DOTALL | re.IGNORECASE
                ).strip()
            
            f.write(json.dumps(response_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to save conversation to JSONL: {e}")


async def run_single_case_translation(case_type, case_name):
    """Run translation for a single case with detailed timing statistics."""
    # Start timing for end-to-end duration
    case_start_time = time.time()
    
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Testing case type: {case_type}")
    logger.info(f"📁 Case name: {case_name}")
    logger.info(f"{'='*60}")

    testcase_src_dir = TESTSET_ROOT_DIR / case_name
    test_work_dir = WORK_DIR / case_name
    test_work_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize timing statistics
    timing_stats = {
        "case_type": case_type,
        "case_name": case_name,
        "start_time": datetime.fromtimestamp(case_start_time).isoformat(),
        "llm_rounds": [],
        "test_rounds": [],
        "total_llm_time": 0,
        "total_test_time": 0,
    }

    # Copy necessary files
    files_to_copy = [DIR_TORCH_ / "ref.py", DIR_CUDA_ / "kernel.cu",
                     f"check_cuda{CHECK_SUFFIX}.py", f"check_triton{CHECK_SUFFIX}.py", "get_data.py"]

    for file_path in files_to_copy:
        src_file = testcase_src_dir / file_path
        dst_file = test_work_dir / file_path

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            logger.debug(f"Copy {file_path} to {test_work_dir}")
        else:
            logger.warning(f"Source file {src_file} does not exist")

    # Read CUDA code
    cuda_file_path = test_work_dir / DIR_CUDA_ / "kernel.cu"
    if not cuda_file_path.exists():
        logger.error(f"CUDA file not found: {cuda_file_path}")
        return False

    with open(cuda_file_path, "r") as f:
        cuda_code = f.read()

    # 创建累积的对话历史，添加系统提示
    conversation_history = [
        make_openai_message_system(
            'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming. You help convert CUDA kernels to Triton kernels while maintaining correctness and performance.'),
        make_openai_message_user(
            simple_initial_prompt.format(cuda_code=cuda_code))
    ]

    # Get API parameters based on model type and call LLM with retry logic
    retry_count = 0
    resp_content = None
    pass_id = 1  # Currently pass@1, will support pass@n in future
    round_id = 1  # Initial generation is round 1
    
    # Time initial LLM call (Round 1)
    llm_round_start = time.time()
    
    actual_model_used = None  # Track actual model used
    
    while retry_count <= args.max_retries:
        try:
            api_params = get_api_param(conversation_history, model_name)
            resp_content, actual_model_used = await async_openai_llm_call(async_client, api_params)
            
            # If API doesn't return model info, use the requested model_name
            if not actual_model_used:
                actual_model_used = model_name
            
            # Log model used (especially useful for openrouter/auto)
            if actual_model_used != model_name:
                logger.info(f"🤖 Model used: {actual_model_used} (requested: {model_name})")
            else:
                logger.debug(f"🤖 Model used: {actual_model_used}")
            
            # Save conversation (including retries)
            await save_llm_conversation(
                test_work_dir=test_work_dir,
                pass_id=pass_id,
                round_id=round_id,
                attempt_id=retry_count,
                messages=conversation_history,
                full_response=resp_content,
                model_used=actual_model_used
            )
            
            # Check if we got valid code (not just the original response due to missing code block)
            triton_code = get_last_code_block(resp_content)
            if triton_code == resp_content:
                # This means get_last_code_block couldn't find a code block and returned the full response
                # This should be retried, not counted as a failed attempt
                if retry_count < args.max_retries:
                    retry_count += 1
                    wait_time = args.retry_wait // 2  # Shorter wait for format issues
                    logger.warning(f"🔄 No code block found in response (retry {retry_count}/{args.max_retries})")
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    
                    # Wait with countdown
                    for remaining in range(wait_time, 0, -1):
                        if remaining % 5 == 0 or remaining <= 3:
                            logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                        await asyncio.sleep(1)
                    
                    logger.info(f"🔄 Retrying API call for better code format...")
                    continue
                else:
                    logger.error(f"❌ Max retries exceeded - no valid code block found")
                    return False, None
            
            if retry_count > 0:
                logger.info(f"✅ Successfully generated code after {retry_count} retries")
            logger.debug(f"Successfully generated initial Triton code")
            
            # Record LLM round timing
            llm_round_end = time.time()
            llm_round_duration = llm_round_end - llm_round_start
            timing_stats["llm_rounds"].append({
                "round": 1,
                "start_time": datetime.fromtimestamp(llm_round_start).isoformat(),
                "end_time": datetime.fromtimestamp(llm_round_end).isoformat(),
                "duration_sec": round(llm_round_duration, 3),
                "retries": retry_count
            })
            timing_stats["total_llm_time"] += llm_round_duration
            break
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if this is a retryable error
            if is_retryable_error(error_msg) and retry_count < args.max_retries:
                retry_count += 1
                wait_time = args.retry_wait
                logger.warning(f"🔄 API overload detected (retry {retry_count}/{args.max_retries}): {error_msg}")
                logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                
                # Wait with countdown (only show every 10 seconds to avoid spam)
                for remaining in range(wait_time, 0, -1):
                    if remaining % 10 == 0 or remaining <= 5:
                        logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                    await asyncio.sleep(1)
                
                logger.info(f"🔄 Retrying API call (attempt {retry_count + 1})...")
                continue
            else:
                # Non-retryable error or max retries exceeded
                if retry_count >= args.max_retries:
                    logger.error(f"❌ Max retries ({args.max_retries}) exceeded for initial generation")
                logger.error(f"Failed to generate initial Triton code: {error_msg}")
                return False, None
    
    if resp_content is None:
        logger.error(f"Failed to generate initial Triton code after all retries")
        return False, None
    
    # 将LLM的回复添加到对话历史中
    conversation_history.append(make_openai_message_assistant(resp_content))

    TRITON_DIR = test_work_dir / DIR_TRITON_
    TRITON_DIR.mkdir(parents=True, exist_ok=True)
    final_triton_code = get_last_code_block(resp_content)
    with open(TRITON_DIR / "kernel.py", "w") as f:
        f.write(final_triton_code)

    logger.info(f"Triton code generated successfully")

    # Save initial conversation history (round 1)
    save_conversation_history(conversation_history,
                              test_work_dir, round_num=1, timestamp=TIMESTAMP)

    # 开始自动化测试和修复流程
    logger.info(f"Starting automated testing and fixing process...")

    success, rounds = await run_testing_loop(conversation_history, test_work_dir, timing_stats, pass_id=pass_id)
    
    # Calculate end-to-end timing
    case_end_time = time.time()
    case_wall_duration = case_end_time - case_start_time
    
    # Calculate effective processing time (LLM + Test execution, excluding GPU wait)
    effective_processing_time = timing_stats["total_llm_time"] + timing_stats["total_test_time"]
    
    # Finalize timing stats
    timing_stats["end_time"] = datetime.fromtimestamp(case_end_time).isoformat()
    timing_stats["wall_clock_duration_sec"] = round(case_wall_duration, 3)  # Total wall-clock time
    timing_stats["effective_duration_sec"] = round(effective_processing_time, 3)  # LLM + Test only
    timing_stats["total_llm_time_sec"] = round(timing_stats["total_llm_time"], 3)
    timing_stats["total_test_time_sec"] = round(timing_stats["total_test_time"], 3)
    timing_stats["overhead_sec"] = round(case_wall_duration - effective_processing_time, 3)  # Other overhead
    timing_stats["success"] = success
    timing_stats["final_round"] = rounds
    timing_stats["timestamp"] = TIMESTAMP
    timing_stats["model_name"] = model_name
    # If actual_model_used is still None (e.g., all retries failed), use model_name as fallback
    timing_stats["actual_model_used"] = actual_model_used if actual_model_used else model_name
    timing_stats["testset"] = args.testset
    timing_stats["use_nvgpu"] = args.use_nvgpu
    timing_stats["concurrency"] = args.concurrency
    timing_stats["gpu_server"] = args.nvgpu_server if args.use_nvgpu else None
    
    # Write to JSONL log immediately
    await write_jsonl_log(timing_stats)
    
    logger.info(f"⏱️  Timing: Wall={case_wall_duration:.1f}s, Effective(LLM+Test)={effective_processing_time:.1f}s (LLM={timing_stats['total_llm_time']:.1f}s, Test={timing_stats['total_test_time']:.1f}s)")
    
    # Log final model used info (only if different from requested)
    final_model = actual_model_used if actual_model_used else model_name
    if final_model != model_name:
        logger.info(f"🤖 Final model used: {final_model} (requested: {model_name})")
    
    return success, rounds


def get_last_code_block(resp_content):
    """Extract the last code block from response content, preferring the last ```python ... ``` block if present.

    Handles cases where ```python tags may not be properly closed.
    """
    resp_content = re.sub(r'<thought>.*?</thought>', '',
                          resp_content, flags=re.DOTALL)

    # 首先找到所有的```python开始标签
    python_starts = []
    for match in re.finditer(r'```python\b', resp_content, re.IGNORECASE):
        python_starts.append(match.end())

    if python_starts:
        # 找到最后一个python开始位置
        last_python_start = python_starts[-1]

        # 从最后一个python标签开始，找到下一个```结束标签或字符串结尾
        remaining_content = resp_content[last_python_start:]

        # 查找结束标签
        end_match = re.search(r'```', remaining_content)
        if end_match:
            # 找到了结束标签
            extracted_code = remaining_content[:end_match.start()].strip()
        else:
            # 没有找到结束标签，取到字符串结尾
            extracted_code = remaining_content.strip()

        return extracted_code

    # 如果没有找到python标签，回退到原来的逻辑
    code_block_pattern = r'```(\w+)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, resp_content, re.DOTALL)

    if matches:
        # 取最后一个任意代码块
        extracted_code = matches[-1][1].strip()
    else:
        extracted_code = resp_content
        logger.debug("Cannot find code block, using original content (will retry)")

    return extracted_code


def save_conversation_history(conversation_history, test_work_dir, round_num=None, timestamp=TIMESTAMP):
    """Save the conversation history to a JSON file for debugging and reference."""
    try:
        logs_dir = test_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if round_num is not None:
            conversation_file = logs_dir / \
                f"conversation_history_round_{round_num}.json"
        else:
            conversation_file = logs_dir / \
                f"conversation_history_{timestamp}.json"

        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)

        logger.debug(f"Conversation history saved to {conversation_file}")
        return str(conversation_file)
    except Exception as e:
        logger.warning(f"Failed to save conversation history: {e}")
        return None


async def run_test_round(round_num, test_work_dir, timing_stats=None):
    """Run a single test round and capture all output with timing."""
    logger.debug(f"Running test round {round_num}...")

    # Backup current kernel
    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
    backup_path = test_work_dir / DIR_TRITON_ / f"kernel_v{round_num}.py"
    shutil.copy(kernel_path, backup_path)
    logger.debug(f"Backed up kernel to kernel_v{round_num}.py")

    # Create logs directory if it doesn't exist
    logs_dir = test_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"triton_test_round_{round_num}.log"
    
    # Use NVGPU server if enabled
    if args.use_nvgpu and NVGPU_AVAILABLE:
        return await run_test_round_nvgpu(round_num, test_work_dir, log_file, timing_stats)
    else:
        return await run_test_round_local(round_num, test_work_dir, log_file, timing_stats)


async def run_test_round_local(round_num, test_work_dir, log_file, timing_stats=None):
    """Run test locally using subprocess with timing."""
    cmd = [sys.executable, f"check_triton{CHECK_SUFFIX}.py"]
    if args.no_perf:
        cmd.append("--no-perf")
    
    # Start timing
    test_start_time = time.time()
    
    try:
        # Create process with asyncio
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(test_work_dir)
        )

        # Wait for process with timeout
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), 
                timeout=10000
            )
            stdout = stdout_data.decode('utf-8')
            stderr = stderr_data.decode('utf-8')
            returncode = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            error_msg = f"Test round {round_num} timed out"
            with open(log_file, 'w') as f:
                f.write(f"=== Test Round {round_num} ===\n")
                f.write(f"ERROR: {error_msg}\n")
            return False, "", error_msg, log_file

        # End timing
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        
        # Record test timing (actual execution time)
        if timing_stats is not None:
            timing_stats["test_rounds"].append({
                "round": round_num,
                "start_time": datetime.fromtimestamp(test_start_time).isoformat(),
                "end_time": datetime.fromtimestamp(test_end_time).isoformat(),
                "duration_sec": round(test_duration, 3),
                "success": returncode == 0 and ("PASSED" in stdout),
                "execution_mode": "local"
            })
            timing_stats["total_test_time"] += test_duration

        # Write full output to log file
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} (Local) ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {returncode}\n")
            f.write(f"Execution time: {test_duration:.3f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)
            f.write(f"\n=== END OF LOG ===\n")

        logger.debug(f"Test output saved to {log_file}")

        # Check if test passed
        success = returncode == 0 and ("PASSED" in stdout)
        return success, stdout, stderr, log_file

    except Exception as e:
        error_msg = f"Test round {round_num} failed: {str(e)}"
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file


async def run_test_round_nvgpu(round_num, test_work_dir, log_file, timing_stats=None):
    """Run test via NVGPU server with timing (excluding queue wait time)."""
    try:
        # Initialize NVGPU client
        nvgpu_client = NVGPUClient(args.nvgpu_server)
        
        # Check server health
        if not nvgpu_client.health_check():
            logger.error(f"NVGPU server at {args.nvgpu_server} is not responding")
            return False, "", "NVGPU server not responding", log_file
        
        logger.debug(f"Connected to NVGPU server at {args.nvgpu_server}")
        
        # Prepare script path and arguments
        script_path = str((test_work_dir / f"check_triton{CHECK_SUFFIX}.py").absolute())
        task_args = ["--no-perf"] if args.no_perf else []
        
        # Submit task
        logger.info(f"Submitting task to NVGPU server (GPU: {args.nvgpu_gpu or 'auto'})")
        task_id = nvgpu_client.submit_task_in_script_dir(
            script_path=script_path,
            task_type=args.nvgpu_task_type,
            # work_dir=str(test_work_dir.absolute()),
            args=task_args,
            gpu_id=args.nvgpu_gpu
        )
        
        logger.info(f"Task submitted: {task_id}")
        
        # Wait for task completion with progress updates
        start_time = time.time()
        last_status = None
        
        while True:
            result = nvgpu_client.get_task(task_id)
            current_status = result.status
            
            if current_status != last_status:
                elapsed = time.time() - start_time
                logger.info(f"Task {task_id[:8]}: {current_status} (elapsed: {elapsed*1000:.0f}ms)")
                last_status = current_status
            
            if current_status in ["completed", "failed", "cancelled"]:
                break
            
            await asyncio.sleep(2)  # Poll every 2 seconds
        
        elapsed_total = time.time() - start_time
        
        # Calculate actual execution time (from task start to end) - THIS is what we want!
        execution_time = None
        waiting_time = None
        if result.start_time and result.end_time:
            from datetime import datetime
            # Parse timestamps if they are strings
            if isinstance(result.start_time, str):
                start_dt = datetime.fromisoformat(result.start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(result.end_time.replace('Z', '+00:00'))
            else:
                start_dt = result.start_time
                end_dt = result.end_time
            execution_time = (end_dt - start_dt).total_seconds()
            waiting_time = elapsed_total - execution_time
        
        # Record test timing (only actual execution time, not waiting time)
        if timing_stats is not None and execution_time is not None:
            timing_stats["test_rounds"].append({
                "round": round_num,
                "start_time": result.start_time if isinstance(result.start_time, str) else result.start_time.isoformat(),
                "end_time": result.end_time if isinstance(result.end_time, str) else result.end_time.isoformat(),
                "duration_sec": round(execution_time, 3),
                "waiting_time_sec": round(waiting_time, 3) if waiting_time else 0,
                "total_elapsed_sec": round(elapsed_total, 3),
                "success": result.status == "completed" and result.exit_code == 0,
                "execution_mode": "nvgpu",
                "gpu_id": getattr(result, 'gpu_id', getattr(result, 'assigned_gpu', 'auto')),
                "task_id": task_id
            })
            # Only add execution time, NOT waiting time!
            timing_stats["total_test_time"] += execution_time
        
        logger.info(f"Task finished in {elapsed_total*1000:.0f}ms with status: {result.status}")
        if execution_time is not None:
            logger.info(f"  └─ Waiting time: {waiting_time*1000:.0f}ms, Execution time: {execution_time*1000:.0f}ms")
        
        # Get task log files from server using nvgpu_client API
        stdout_content = ""
        stderr_content = ""
        
        try:
            # Use nvgpu_client to fetch full logs
            stdout_content = nvgpu_client.get_full_task_log(task_id, log_type="stdout")
            logger.debug(f"Retrieved stdout ({len(stdout_content)} bytes)")
        except Exception as log_e:
            logger.warning(f"Failed to fetch stdout for task {task_id}: {log_e}")
            stdout_content = f"Error fetching stdout: {log_e}"
        
        try:
            stderr_content = nvgpu_client.get_full_task_log(task_id, log_type="stderr")
            logger.debug(f"Retrieved stderr ({len(stderr_content)} bytes)")
        except Exception as log_e:
            logger.warning(f"Failed to fetch stderr for task {task_id}: {log_e}")
            stderr_content = f"Error fetching stderr: {log_e}"
        
        # Write local log file
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} (NVGPU) ===\n")
            f.write(f"Task ID: {task_id}\n")
            # Use assigned_gpu from result (new API returns this as gpu_id attribute)
            # Note: Use 'is not None' check because GPU ID 0 is valid but falsy
            assigned_gpu = getattr(result, 'gpu_id', None)
            if assigned_gpu is None:
                assigned_gpu = getattr(result, 'assigned_gpu', None)
            f.write(f"GPU: {assigned_gpu if assigned_gpu is not None else (args.nvgpu_gpu or 'auto-assigned')}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Exit code: {result.exit_code}\n")
            f.write(f"\n=== Timing Information ===\n")
            f.write(f"Total elapsed time (submit to finish): {elapsed_total*1000:.0f}ms\n")
            if execution_time is not None:
                f.write(f"Waiting time (queue): {waiting_time*1000:.0f}ms\n")
                f.write(f"Execution time (actual run): {execution_time*1000:.0f}ms\n")
            if hasattr(result, 'submit_time') and result.submit_time:
                f.write(f"Submit time: {result.submit_time}\n")
            if hasattr(result, 'start_time') and result.start_time:
                f.write(f"Start time: {result.start_time}\n")
            if hasattr(result, 'end_time') and result.end_time:
                f.write(f"End time: {result.end_time}\n")
            f.write(f"\nServer log: {result.log_file}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout_content)
            f.write("\n=== STDERR ===\n")
            f.write(stderr_content)
            f.write(f"\n=== END OF LOG ===\n")
        
        logger.debug(f"Test output saved to {log_file}")
        
        # Check if test passed
        success = result.status == "completed" and result.exit_code == 0 and "PASSED" in stdout_content
        
        return success, stdout_content, stderr_content, log_file
        
    except Exception as e:
        error_msg = f"NVGPU execution failed: {str(e)}"
        logger.error(error_msg)
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file


async def get_feedback_from_llm(round_num, error_output, stderr_output, test_work_dir, conversation_history, timing_stats=None, pass_id=1):
    """Get feedback from LLM to fix the triton kernel using cumulative conversation history."""
    llm_round_start = time.time()
    
    try:
        # Prepare feedback prompt
        error_info = f"Round {round_num} Test Output:\n{error_output}\n\nStderr:\n{stderr_output}"
        traceback_info = stderr_output if stderr_output else "No traceback available"

        prompt = feedback_prompt.format(
            error_info=error_info,
            traceback_info=traceback_info
        )

        logger.debug(f"Requesting LLM feedback for round {round_num}...")

        # Add the current kernel code and error feedback to conversation history
        conversation_history.append(make_openai_message_user(prompt))

        # Call LLM with retry logic
        retry_count = 0
        fixed_code_response = None
        feedback_round_id = round_num + 1  # Feedback produces next round code
        feedback_model_used = None  # Track actual model used in feedback
        
        while retry_count <= args.max_retries:
            try:
                # Get API parameters and call LLM
                api_params = get_api_param(conversation_history, model_name)
                fixed_code_response, feedback_model_used = await async_openai_llm_call(async_client, api_params)
                
                # If API doesn't return model info, use the requested model_name
                if not feedback_model_used:
                    feedback_model_used = model_name
                
                # Log model used (especially useful for openrouter/auto)
                if feedback_model_used != model_name:
                    logger.info(f"🤖 Feedback model used: {feedback_model_used} (requested: {model_name})")
                else:
                    logger.debug(f"🤖 Feedback model used: {feedback_model_used}")
                
                # Save conversation (including retries)
                await save_llm_conversation(
                    test_work_dir=test_work_dir,
                    pass_id=pass_id,
                    round_id=feedback_round_id,
                    attempt_id=retry_count,
                    messages=conversation_history,
                    full_response=fixed_code_response,
                    model_used=feedback_model_used
                )
                
                # Check if we got valid code (not just the original response due to missing code block)
                fixed_code = get_last_code_block(fixed_code_response)
                if fixed_code == fixed_code_response:
                    # This means get_last_code_block couldn't find a code block and returned the full response
                    # This should be retried, not counted as a failed attempt
                    if retry_count < args.max_retries:
                        retry_count += 1
                        wait_time = args.retry_wait // 2  # Shorter wait for format issues
                        logger.warning(f"🔄 No code block found in feedback response (retry {retry_count}/{args.max_retries})")
                        logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                        
                        # Wait with countdown
                        for remaining in range(wait_time, 0, -1):
                            if remaining % 5 == 0 or remaining <= 3:
                                logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                            await asyncio.sleep(1)
                        
                        logger.info(f"🔄 Retrying feedback API call for better code format...")
                        continue
                    else:
                        logger.error(f"❌ Max retries exceeded - no valid code block found in feedback")
                        return None
                
                if retry_count > 0:
                    logger.info(f"✅ Successfully generated feedback after {retry_count} retries")
                logger.debug(f"Successfully got LLM feedback for round {round_num}")
                
                # Record LLM feedback round timing
                llm_round_end = time.time()
                llm_round_duration = llm_round_end - llm_round_start
                if timing_stats is not None:
                    timing_stats["llm_rounds"].append({
                        "round": round_num + 1,  # This is the feedback for next round
                        "start_time": datetime.fromtimestamp(llm_round_start).isoformat(),
                        "end_time": datetime.fromtimestamp(llm_round_end).isoformat(),
                        "duration_sec": round(llm_round_duration, 3),
                        "retries": retry_count
                    })
                    timing_stats["total_llm_time"] += llm_round_duration
                break
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a retryable error
                if is_retryable_error(error_msg) and retry_count < args.max_retries:
                    retry_count += 1
                    wait_time = args.retry_wait
                    logger.warning(f"🔄 API overload detected in feedback (retry {retry_count}/{args.max_retries}): {error_msg}")
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    
                    # Wait with countdown (only show every 10 seconds to avoid spam)
                    for remaining in range(wait_time, 0, -1):
                        if remaining % 10 == 0 or remaining <= 5:
                            logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                        await asyncio.sleep(1)
                    
                    logger.info(f"🔄 Retrying feedback API call (attempt {retry_count + 1})...")
                    continue
                else:
                    # Non-retryable error or max retries exceeded
                    if retry_count >= args.max_retries:
                        logger.error(f"❌ Max retries ({args.max_retries}) exceeded for feedback round {round_num}")
                    logger.error(f"Failed to get LLM feedback for round {round_num}: {error_msg}")
                    return None
        
        if fixed_code_response is None:
            logger.error(f"Failed to get LLM feedback after all retries")
            return None

        # Add LLM response to conversation history
        conversation_history.append(make_openai_message_assistant(fixed_code_response))

        fixed_code = get_last_code_block(fixed_code_response)
        return fixed_code

    except Exception as e:
        logger.error(f"Failed to get LLM feedback: {str(e)}")
        return None


async def run_testing_loop(conversation_history, test_work_dir, timing_stats=None, pass_id=1):
    """Run the testing and fixing loop with timing statistics."""
    for round_num in range(1, MAX_ROUNDS + 1):
        success, stdout, stderr, log_file = await run_test_round(
            round_num, test_work_dir, timing_stats)

        if success:
            logger.info(
                f"✅ Test round {round_num} PASSED! Triton kernel is working correctly.")
            logger.info(f"Final results saved in {log_file}")

            return True, round_num
        else:
            logger.info(f"❌ Test round {round_num} FAILED.")
            logger.debug(f"Error details saved in {log_file}")

            if round_num < MAX_ROUNDS:
                logger.info(f"🔧 Attempting to fix with LLM feedback...")

                # Get feedback from LLM using cumulative conversation history
                # 这将产生第(round_num+1)轮对话
                fixed_code = await get_feedback_from_llm(
                    round_num, stdout, stderr, test_work_dir, conversation_history, timing_stats, pass_id)

                if fixed_code:
                    # Save the fixed code
                    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
                    with open(kernel_path, 'w') as f:
                        f.write(fixed_code)
                    logger.debug(f"Updated kernel.py with LLM feedback")

                    # 保存第(round_num+1)轮对话历史
                    save_conversation_history(
                        conversation_history, test_work_dir, round_num + 1, timestamp=TIMESTAMP)
                else:
                    logger.error(
                        f"Failed to get valid LLM feedback for round {round_num}")
                    break
            else:
                logger.error(
                    f"Maximum rounds ({MAX_ROUNDS}) reached. Manual intervention required.")
                break

    return False, MAX_ROUNDS


async def test_cases():
    """Test cases with async concurrency control and real-time JSONL logging."""
    logger.info(f"🚀 Starting batch testing with async execution...")
    logger.info(f"📋 Available case types: {list(AVAILABLE_CASES.keys())}")
    logger.info(f"📁 Work directory: {WORK_DIR}")
    logger.info(f"📊 Real-time statistics: {JSONL_LOG_FILE}")
    
    # Initialize JSONL log file (clear if exists)
    try:
        with open(JSONL_LOG_FILE, 'w', encoding='utf-8') as f:
            # Write metadata header as first line
            metadata = {
                "_type": "metadata",
                "model_name": model_name,
                "testset": args.testset,
                "timestamp": TIMESTAMP,
                "max_rounds": MAX_ROUNDS,
                "concurrency": args.concurrency,
                "use_nvgpu": args.use_nvgpu,
                "nvgpu_server": args.nvgpu_server if args.use_nvgpu else None,
                "no_perf": args.no_perf,
                "created_at": datetime.now().isoformat()
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        logger.debug(f"Initialized JSONL log file: {JSONL_LOG_FILE}")
    except Exception as e:
        logger.warning(f"Failed to initialize JSONL log file: {e}")
    
    # Log configuration summary
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   - Model: {model_name}")
    logger.info(f"   - Testset: {args.testset}")
    logger.info(f"   - Max rounds: {MAX_ROUNDS}")
    logger.info(f"   - Console output: {CONSOLE_OUTPUT}")
    logger.info(f"   - First only: {args.first_only}")
    logger.info(f"   - Skip performance: {args.no_perf}")
    logger.info(f"   - Retry wait time: {args.retry_wait}s")
    logger.info(f"   - Max retries: {args.max_retries}")
    logger.info(f"   - Specific case types: {args.case_types if args.case_types else 'All'}")
    logger.info(f"   - Use NVGPU: {args.use_nvgpu}")
    logger.info(f"   - Concurrency: {args.concurrency}")
    if args.use_nvgpu:
        logger.info(f"   - NVGPU server: {args.nvgpu_server}")
        logger.info(f"   - NVGPU GPU: {args.nvgpu_gpu or 'auto-assign'}")
        logger.info(f"   - NVGPU task type: {args.nvgpu_task_type}")

    # 使用详细的结果存储结构
    detailed_results = {}  # case_type -> list of case results
    all_case_results = []  # 所有单个case的结果列表

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def run_case_with_semaphore(case_type, case_name_single):
        """Run a single case with semaphore control."""
        async with semaphore:
            try:
                logger.info(
                    f"{'🔹'*20} Starting {case_type}/{case_name_single} {'🔹'*20}")
                success, rounds = await run_single_case_translation(
                    case_type, case_name_single)

                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": success,
                    "rounds": rounds
                }

                if success:
                    status = f"✅ SUCCESS (Round {rounds})"
                else:
                    status = f"❌ FAILED (Round {rounds})"
                logger.info(f"{'🔹'*15} {case_type}/{case_name_single}: {status} {'🔹'*15}")
                
                return case_result

            except Exception as e:
                logger.error(f"Error in {case_type}/{case_name_single}: {e}")
                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": False,
                    "rounds": None,
                    "error": str(e)
                }
                return case_result
    
    # Prepare all tasks
    tasks = []
    for case_type, case_name in AVAILABLE_CASES.items():
        if not (isinstance(case_name, tuple) or isinstance(case_name, list)):
            case_name = [case_name]

        if case_type not in detailed_results:
            detailed_results[case_type] = []

        for case_name_single in case_name:
            task = run_case_with_semaphore(case_type, case_name_single)
            tasks.append((case_type, task))
    
    # Execute all tasks concurrently
    logger.info(f"🚀 Launching {len(tasks)} tasks with max concurrency {args.concurrency}...")
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    # Organize results and handle exceptions
    for (case_type, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            error_result = {
                "case_type": case_type,
                "case_name": "unknown",
                "success": False,
                "rounds": None,
                "error": str(result)
            }
            detailed_results[case_type].append(error_result)
            all_case_results.append(error_result)
        else:
            detailed_results[case_type].append(result)
            all_case_results.append(result)

    # Print detailed summary
    logger.info(f"{'='*80}")
    logger.info(f"📊 DETAILED BATCH TESTING SUMMARY")
    logger.info(f"{'='*80}")

    total_success = 0
    total_cases = 0

    for case_type, case_results in detailed_results.items():
        case_type_success = sum(
            1 for result in case_results if result["success"])
        case_type_total = len(case_results)
        total_success += case_type_success
        total_cases += case_type_total

        # Case type level summary
        case_type_status = "✅" if case_type_success == case_type_total else "❌" if case_type_success == 0 else "⚠️ "
        logger.info(
            f"{case_type_status} {case_type:<15} ({case_type_success}/{case_type_total})")

        # Individual case details
        for result in case_results:
            if result["success"]:
                status = f"  ✅ (Round {result['rounds']})"
            else:
                rounds_info = f"Round {result['rounds']}" if result.get(
                    'rounds') is not None else "Error"
                status = f"  ❌ ({rounds_info})"

            error_info = f" - {result.get('error', '')}" if not result["success"] and 'error' in result else ""
            logger.info(f"{status} {result['case_name']}{error_info}")
        logger.info("")

    # 轮次分布统计
    rounds_distribution = {}
    successful_cases = [
        result for result in all_case_results if result["success"]]

    for result in successful_cases:
        round_num = result["rounds"]
        rounds_distribution[round_num] = rounds_distribution.get(
            round_num, 0) + 1

    logger.info(f"📊 Success rounds distribution:")
    for round_num in sorted(rounds_distribution.keys()):
        count = rounds_distribution[round_num]
        logger.info(f"   Round {round_num}: {count} cases")

    avg_rounds = sum(result["rounds"] for result in successful_cases) / \
        len(successful_cases) if successful_cases else 0
    logger.info(f"📈 Average rounds for successful cases: {avg_rounds:.2f}")

    logger.info(f"{'='*80}")
    logger.info(
        f"🏆 OVERALL TOTAL: {total_success}/{total_cases} individual cases succeeded")
    logger.info(
        f"📊 Case type success rate: {sum(1 for case_type, results in detailed_results.items() if all(r['success'] for r in results))}/{len(detailed_results)} case types fully passed")
    logger.info(f"📋 Log file: {log_file}")
    logger.info(f"📊 Real-time statistics: {JSONL_LOG_FILE}")
    logger.info(f"{'='*80}")
    
    # Write final summary to JSONL
    final_summary = {
        "_type": "summary",
        "timestamp": TIMESTAMP,
        "total_cases": total_cases,
        "successful_cases": total_success,
        "failed_cases": total_cases - total_success,
        "success_rate": total_success / total_cases if total_cases > 0 else 0,
        "case_types_total": len(detailed_results),
        "case_types_fully_passed": sum(1 for case_type, results in detailed_results.items() if all(r['success'] for r in results)),
        "rounds_distribution": rounds_distribution,
        "avg_rounds": avg_rounds,
        "completed_at": datetime.now().isoformat()
    }
    await write_jsonl_log(final_summary)

    return {"detailed_results": detailed_results, "all_case_results": all_case_results}


if __name__ == "__main__":
    """
    Examples of usage:
    
    # Test all cases with iterative fixing using default (xpiler) testset with NVGPU
    python llm_trans.py
    
    # Test with xpiler testset
    python llm_trans.py --testset xpiler
    
    # Test with different model
    python llm_trans.py --model claude --testset xpiler
    
    # Test only specific case types
    python llm_trans.py --case-types conv2d gemm layernorm
    
    # Test only first case from each type with no console output
    python llm_trans.py --first-only --no-console --testset leetcuda_dynamic
    
    # Test with different model and skip performance testing
    python llm_trans.py --model qwen --case-types add --no-perf --testset leetcuda_dynamic
    
    # Test with custom retry settings (30s wait, max 3 retries)
    python llm_trans.py --retry-wait 30 --max-retries 3
    
    # Test with fast retries for development
    python llm_trans.py --first-only --retry-wait 10 --max-retries 2
    
    # Use NVGPU server for task execution (now default)
    python llm_trans.py --nvgpu-server http://localhost:8080
    
    # Disable NVGPU and use local execution
    python llm_trans.py --no-nvgpu
    
    # Use NVGPU with specific GPU
    python llm_trans.py --nvgpu-gpu 0
    
    # Use NVGPU for performance testing
    python llm_trans.py --nvgpu-task-type performance
    
    # Combined: NVGPU + first case + specific GPU + custom concurrency
    python llm_trans.py --first-only --nvgpu-gpu 1 --case-types add --concurrency 5
    
    # High concurrency with async execution (default: 10)
    python llm_trans.py --concurrency 20
    """
    
    # Validate configuration
    if not AVAILABLE_CASES:
        logger.error("No cases available for testing. Check your case types filter.")
        sys.exit(1)
    
    # Test all cases using async iterative fixing methodology
    asyncio.run(test_cases())
