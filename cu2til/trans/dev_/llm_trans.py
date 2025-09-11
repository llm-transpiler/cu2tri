from case_config import XPILER_ALL_CASES
from openai import OpenAI
import os
import shutil
from pathlib import Path
from cu2til.prompt.cuda2triton import simple_initial_prompt, feedback_prompt
import dotenv
import subprocess
import sys
import json
from datetime import datetime
from cu2til.trans.dev_.llm import get_api_param_openai_default
from llm import openai_llm_call, CallingIdentifier, get_api_params_method, make_openai_message_system, make_openai_message_user, make_openai_message_assistant
# 执行自动化测试和修复
dotenv.load_dotenv()
# run_model = "qwen"
# run_model = "gemini"
# run_model = "deepseek"
# run_model = "claude"
run_model = "gpt"
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

print(f"Using {run_model} model: {model_name}")
# Import case configuration

DIR_CUDA_ = Path("cuda_")
DIR_TORCH_ = Path("torch_")
DIR_TRITON_ = Path("triton_")
TESTSET_ROOT_DIR = Path("/workspace/cu2til/cases/xpiler")
WORK_DIR = Path(__file__).parent.absolute()
TEMPERATURE = 0.35
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
# TIMESTAMP = "20250910_210335"
WORK_DIR = Path(__file__).parent.absolute() / "".join(c if c.isalnum()
                                                      else "_" for c in model_name.split("/")[-1].lower()) / TIMESTAMP
MAX_ROUNDS = 5


def get_first_case_from_each_type():
    """Extract the first case from each case type in XPILER_ALL_CASES."""
    first_cases = {}
    for case_type, cases in XPILER_ALL_CASES.items():
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
        if case_type in XPILER_ALL_CASES:
            failed_cases[case_type] = XPILER_ALL_CASES[case_type]
        else:
            print(
                f"⚠️ Warning: case_type '{case_type}' not found in XPILER_ALL_CASES")

    return failed_cases


# Configuration - can be changed to test different cases
# AVAILABLE_CASES = get_first_case_from_each_type()
AVAILABLE_CASES = XPILER_ALL_CASES
# AVAILABLE_CASES = get_failed_cases()


def run_single_case_translation(case_type, case_name):
    """Run translation for a single case."""
    print(f"\n{'='*60}")
    print(f"🎯 Testing case type: {case_type}")
    print(f"📁 Case name: {case_name}")
    print(f"{'='*60}")

    testcase_src_dir = TESTSET_ROOT_DIR / case_name
    test_work_dir = WORK_DIR / case_name
    test_work_dir.mkdir(parents=True, exist_ok=True)

    # Copy necessary files
    files_to_copy = [DIR_TORCH_ / "ref.py", DIR_CUDA_ / "kernel.cu",
                     "check_cuda.py", "check_triton.py", "get_data.py"]

    for file_path in files_to_copy:
        src_file = testcase_src_dir / file_path
        dst_file = test_work_dir / file_path

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"✅ Copy {file_path} to {test_work_dir}")
        else:
            print(f"❌ Warning: Source file {src_file} does not exist")

    # Read CUDA code
    cuda_file_path = test_work_dir / DIR_CUDA_ / "kernel.cu"
    if not cuda_file_path.exists():
        print(f"❌ CUDA file not found: {cuda_file_path}")
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

    print(f"✅ Triton code generated successfully")

    # Save initial conversation history (round 1)
    save_conversation_history(conversation_history,
                              test_work_dir, round_num=1, timestamp=TIMESTAMP)

    # 开始自动化测试和修复流程
    print(f"\n🔄 Starting automated testing and fixing process...")

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
        print(f"⚠️ Cannot find code block, using original content")

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

        print(f"📝 Conversation history saved to {conversation_file}")
        return str(conversation_file)
    except Exception as e:
        print(f"⚠️ Failed to save conversation history: {e}")
        return None


def run_test_round(round_num, test_work_dir):
    """Run a single test round and capture all output."""
    print(f"\n🔄 Running test round {round_num}...")

    # Backup current kernel
    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
    backup_path = test_work_dir / DIR_TRITON_ / f"kernel_v{round_num}.py"
    shutil.copy(kernel_path, backup_path)
    print(f"✅ Backed up kernel to kernel_v{round_num}.py")

    # Create logs directory if it doesn't exist
    logs_dir = test_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Run the test and capture output
    log_file = logs_dir / f"triton_test_round_{round_num}.log"
    cmd = [sys.executable, "check_triton.py"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "1":
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

        print(f"📋 Test output saved to {log_file}")

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

        print(f"🤖 Requesting LLM feedback for round {round_num}...")

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
        print(f"❌ Failed to get LLM feedback: {str(e)}")
        return None


def run_testing_loop(conversation_history, test_work_dir):
    """Run the testing and fixing loop."""
    for round_num in range(1, MAX_ROUNDS + 1):
        success, stdout, stderr, log_file = run_test_round(
            round_num, test_work_dir)

        if success:
            print(
                f"✅ Test round {round_num} PASSED! Triton kernel is working correctly.")
            print(f"📊 Final results saved in {log_file}")

            return True, round_num
        else:
            print(f"❌ Test round {round_num} FAILED.")
            print(f"📋 Error details saved in {log_file}")

            if round_num < MAX_ROUNDS:
                print(f"🔧 Attempting to fix with LLM feedback...")

                # Get feedback from LLM using cumulative conversation history
                # 这将产生第(round_num+1)轮对话
                fixed_code = get_feedback_from_llm(
                    round_num, stdout, stderr, test_work_dir, conversation_history)

                if fixed_code:
                    # Save the fixed code
                    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
                    with open(kernel_path, 'w') as f:
                        f.write(fixed_code)
                    print(f"💾 Updated kernel.py with LLM feedback")

                    # 保存第(round_num+1)轮对话历史
                    save_conversation_history(
                        conversation_history, test_work_dir, round_num + 1, timestamp=TIMESTAMP)
                else:
                    print(
                        f"❌ Failed to get valid LLM feedback for round {round_num}")
                    break
            else:
                print(
                    f"❌ Maximum rounds ({MAX_ROUNDS}) reached. Manual intervention required.")
                break

    return False, MAX_ROUNDS


def test_cases():
    """Test the first case from each case type."""
    print(f"🚀 Starting batch testing of first cases from each type...")
    print(f"📋 Available case types: {list(AVAILABLE_CASES.keys())}")

    # 使用详细的结果存储结构
    detailed_results = {}  # case_type -> list of case results
    all_case_results = []  # 所有单个case的结果列表

    for case_type, case_name in AVAILABLE_CASES.items():
        if not (isinstance(case_name, tuple) or isinstance(case_name, list)):
            case_name = [case_name]

        detailed_results[case_type] = []

        for case_name_single in case_name:
            try:
                print(
                    f"\n{'🔹'*20} Starting {case_type}/{case_name_single} {'🔹'*20}")
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
                print(f"{'🔹'*15} {case_type}/{case_name_single}: {status} {'🔹'*15}")

            except Exception as e:
                print(f"❌ Error in {case_type}/{case_name_single}: {e}")
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
    print(f"\n{'='*80}")
    print(f"📊 DETAILED BATCH TESTING SUMMARY")
    print(f"{'='*80}")

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
        print(
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
            print(f"{status} {result['case_name']}{error_info}")
        print()

    # 轮次分布统计
    rounds_distribution = {}
    successful_cases = [
        result for result in all_case_results if result["success"]]

    for result in successful_cases:
        round_num = result["rounds"]
        rounds_distribution[round_num] = rounds_distribution.get(
            round_num, 0) + 1

    print(f"📊 Success rounds distribution:")
    for round_num in sorted(rounds_distribution.keys()):
        count = rounds_distribution[round_num]
        print(f"   Round {round_num}: {count} cases")

    avg_rounds = sum(result["rounds"] for result in successful_cases) / \
        len(successful_cases) if successful_cases else 0
    print(f"📈 Average rounds for successful cases: {avg_rounds:.2f}")

    print(f"{'='*80}")
    print(
        f"🏆 OVERALL TOTAL: {total_success}/{total_cases} individual cases succeeded")
    print(
        f"📊 Case type success rate: {sum(1 for case_type, results in detailed_results.items() if all(r['success'] for r in results))}/{len(detailed_results)} case types fully passed")
    print(f"{'='*80}")

    return {"detailed_results": detailed_results, "all_case_results": all_case_results}


if __name__ == "__main__":
    # Configuration options:

    # # Option 1: Test a single specific case
    # case_type = "conv2d"  # Change this to test different case types
    # case_name = "conv2d_16_8_8_64_64_2_2_64_2_0"
    # run_single_case_translation(case_type, case_name)
    # exit(0)

    # Option 2: Test all first cases (uncomment to enable)
    test_cases()
