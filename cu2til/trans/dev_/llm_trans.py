import logging
import os
import shutil
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from case_config import XPILER_ALL_CASES, LEETCUDA_DYNAMIC_ALL_CASES
from openai import OpenAI
from cu2til.prompt.cuda2triton import simple_initial_prompt, feedback_prompt
import dotenv
from cu2til.trans.dev_.llm import get_api_param_openai_default
from llm import openai_llm_call, CallingIdentifier, get_api_params_method, make_openai_message_system, make_openai_message_user, make_openai_message_assistant
# 执行自动化测试和修复
dotenv.load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CUDA to Triton translation with iterative fixing')
    parser.add_argument('--model', choices=['qwen', 'gpt', 'gemini', 'deepseek', 'claude'], 
                       default='qwen', help='Model to use for translation')
    parser.add_argument('--console', action='store_true', default=True,
                       help='Output logs to console (default: True)')
    parser.add_argument('--no-console', action='store_true', default=False,
                       help='Disable console output')
    parser.add_argument('--case-types', nargs='*', 
                       help='Specific case types to test (default: all)')
    parser.add_argument('--first-only', action='store_true', default=False,
                       help='Test only the first case from each case type')
    parser.add_argument('--no-perf', action='store_true', default=False,
                       help='Skip performance testing (add --no-perf to check_triton.py)')
    parser.add_argument('--testset', choices=['xpiler', 'leetcuda_dynamic'], 
                       default='leetcuda_dynamic', help='Test set to use (default: xpiler)')
    return parser.parse_args()

# Parse arguments
args = parse_args()
run_model = args.model
get_api_param = get_api_param_openai_default
model_name = "DEFAULT_MODEL_NAME"
if run_model == "qwen":
    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    client = OpenAI(
        base_url='http://10.156.112.253:8000/v1',  # api_base
        api_key="EMPTY"
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OFFICIAL)
elif run_model == "gpt":
    model_name = "openai/gpt-5"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    get_api_param = get_api_params_method(CallingIdentifier.OPENAI_OPENROUTER)
elif run_model == "gemini":
    model_name = "gemini-2.5-pro"
    model_name = "gemini-2.5-flash"
    client = OpenAI(
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
    get_api_param = get_api_params_method(CallingIdentifier.DEEPSEEK_OPENAI)
elif run_model == "claude":
    model_name = "anthropic/claude-sonnet-4"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    get_api_param = get_api_params_method(
        CallingIdentifier.ANTHROPIC_OPENROUTER)
else:
    raise ValueError(f"Unsupported model: {run_model}")

# Constants
DIR_CUDA_ = Path("cuda_")
DIR_TORCH_ = Path("torch_")
DIR_TRITON_ = Path("triton_")
TESTSET_ROOT_DIR = Path(f"/workspace/cu2til/cases/{args.testset}")
TEMPERATURE = 0.35
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MAX_ROUNDS = 5
CONSOLE_OUTPUT = args.console and not args.no_console  # Whether to output to console
CHECK_SUFFIX = "_dynamic" if args.testset.startswith("leetcuda_dynamic") else ""

# Select appropriate case configuration based on testset
ALL_CASES = LEETCUDA_DYNAMIC_ALL_CASES if args.testset.startswith("leetcuda_dynamic") else XPILER_ALL_CASES

# ALL_CASES = {"add": ["add_f16x8_pack"]}
# Setup work directory
model_name_clean = "".join(c if c.isalnum() else "_" for c in model_name.split("/")[-1].lower())
WORK_DIR = Path(__file__).parent.absolute() / f"{model_name_clean}" / TIMESTAMP
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = WORK_DIR / f"{model_name_clean}.log"
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


def run_single_case_translation(case_type, case_name):
    """Run translation for a single case."""
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Testing case type: {case_type}")
    logger.info(f"📁 Case name: {case_name}")
    logger.info(f"{'='*60}")

    testcase_src_dir = TESTSET_ROOT_DIR / case_name
    test_work_dir = WORK_DIR / case_name
    test_work_dir.mkdir(parents=True, exist_ok=True)

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

    # Get API parameters based on model type
    api_params = get_api_param(conversation_history, model_name)

    # Call LLM using the wrapper function
    resp_content = openai_llm_call(client, api_params)

    # 将LLM的回复添加到对话历史中
    conversation_history.append(make_openai_message_assistant(resp_content))

    TRITON_DIR = test_work_dir / DIR_TRITON_
    TRITON_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRITON_DIR / "kernel.py", "w") as f:
        f.write(get_last_code_block(resp_content))

    logger.info(f"Triton code generated successfully")

    # Save initial conversation history (round 1)
    save_conversation_history(conversation_history,
                              test_work_dir, round_num=1, timestamp=TIMESTAMP)

    # 开始自动化测试和修复流程
    logger.info(f"Starting automated testing and fixing process...")

    success, rounds = run_testing_loop(conversation_history, test_work_dir)
    return success, rounds


def get_last_code_block(resp_content):
    """Extract the last code block from response content, preferring the last ```python ... ``` block if present.

    Handles cases where ```python tags may not be properly closed.
    """
    import re
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


def run_test_round(round_num, test_work_dir):
    """Run a single test round and capture all output."""
    logger.debug(f"Running test round {round_num}...")

    # Backup current kernel
    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
    backup_path = test_work_dir / DIR_TRITON_ / f"kernel_v{round_num}.py"
    shutil.copy(kernel_path, backup_path)
    logger.debug(f"Backed up kernel to kernel_v{round_num}.py")

    # Create logs directory if it doesn't exist
    logs_dir = test_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Run the test and capture output
    log_file = logs_dir / f"triton_test_round_{round_num}.log"
    cmd = [sys.executable, f"check_triton{CHECK_SUFFIX}.py"]
    if args.no_perf:
        cmd.append("--no-perf")
    try:
        # Change to target directory for test execution
        original_cwd = os.getcwd()
        os.chdir(test_work_dir)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        # Restore original directory
        os.chdir(original_cwd)

        # Write full output to log file
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== END OF LOG ===\n")

        logger.debug(f"Test output saved to {log_file}")

        # Check if test passed (look for success indicators)
        success = result.returncode == 0 and ("PASSED" in result.stdout)

        return success, result.stdout, result.stderr, log_file

    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)  # Restore directory even on timeout
        error_msg = f"Test round {round_num} timed out after 5 minutes"
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file
    except Exception as e:
        os.chdir(original_cwd)  # Restore directory even on exception
        error_msg = f"Test round {round_num} failed with exception: {str(e)}"
        with open(log_file, 'w') as f:
            f.write(f"=== Test Round {round_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file


def get_feedback_from_llm(round_num, error_output, stderr_output, test_work_dir, conversation_history):
    """Get feedback from LLM to fix the triton kernel using cumulative conversation history."""
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

        # Get API parameters and call LLM
        api_params = get_api_param(conversation_history, model_name)

        # Call LLM using the wrapper function
        fixed_code = openai_llm_call(client, api_params)

        # Add LLM response to conversation history
        conversation_history.append(make_openai_message_assistant(fixed_code))

        fixed_code = get_last_code_block(fixed_code)
        return fixed_code

    except Exception as e:
        logger.error(f"Failed to get LLM feedback: {str(e)}")
        return None


def run_testing_loop(conversation_history, test_work_dir):
    """Run the testing and fixing loop."""
    for round_num in range(1, MAX_ROUNDS + 1):
        success, stdout, stderr, log_file = run_test_round(
            round_num, test_work_dir)

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
                fixed_code = get_feedback_from_llm(
                    round_num, stdout, stderr, test_work_dir, conversation_history)

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


def test_cases():
    """Test the first case from each case type."""
    logger.info(f"🚀 Starting batch testing...")
    logger.info(f"📋 Available case types: {list(AVAILABLE_CASES.keys())}")
    logger.info(f"📁 Work directory: {WORK_DIR}")
    
    # Log configuration summary
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   - Model: {model_name}")
    logger.info(f"   - Testset: {args.testset}")
    logger.info(f"   - Max rounds: {MAX_ROUNDS}")
    logger.info(f"   - Console output: {CONSOLE_OUTPUT}")
    logger.info(f"   - First only: {args.first_only}")
    logger.info(f"   - Skip performance: {args.no_perf}")
    logger.info(f"   - Specific case types: {args.case_types if args.case_types else 'All'}")

    # 使用详细的结果存储结构
    detailed_results = {}  # case_type -> list of case results
    all_case_results = []  # 所有单个case的结果列表

    for case_type, case_name in AVAILABLE_CASES.items():
        if not (isinstance(case_name, tuple) or isinstance(case_name, list)):
            case_name = [case_name]

        detailed_results[case_type] = []

        for case_name_single in case_name:
            try:
                logger.info(
                    f"{'🔹'*20} Starting {case_type}/{case_name_single} {'🔹'*20}")
                success, rounds = run_single_case_translation(
                    case_type, case_name_single)

                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": success,
                    "rounds": rounds
                }
                detailed_results[case_type].append(case_result)
                all_case_results.append(case_result)

                if success:
                    status = f"✅ SUCCESS (Round {rounds})"
                else:
                    status = f"❌ FAILED (Round {rounds})"
                logger.info(f"{'🔹'*15} {case_type}/{case_name_single}: {status} {'🔹'*15}")

            except Exception as e:
                logger.error(f"Error in {case_type}/{case_name_single}: {e}")
                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": False,
                    "rounds": None,
                    "error": str(e)
                }
                detailed_results[case_type].append(case_result)
                all_case_results.append(case_result)

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
    logger.info(f"{'='*80}")

    return {"detailed_results": detailed_results, "all_case_results": all_case_results}


if __name__ == "__main__":
    """
    Examples of usage:
    
    # Test all cases with iterative fixing using default (xpiler) testset
    python llm_trans.py
    
    # Test with leetcuda_dynamic_test testset
    python llm_trans.py --testset leetcuda_dynamic_test
    
    # Test with different model
    python llm_trans.py --model claude --testset xpiler
    
    # Test only specific case types
    python llm_trans.py --case-types conv2d gemm layernorm
    
    # Test only first case from each type with no console output
    python llm_trans.py --first-only --no-console --testset leetcuda_dynamic_test
    
    # Test with different model and skip performance testing
    python llm_trans.py --model qwen --case-types add --no-perf --testset leetcuda_dynamic_test
    """
    
    # Validate configuration
    if not AVAILABLE_CASES:
        logger.error("No cases available for testing. Check your case types filter.")
        sys.exit(1)
    
    # Test all cases using iterative fixing methodology
    test_cases()
