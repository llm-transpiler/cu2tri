import logging
import os
import shutil
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from case_config import XPILER_ALL_CASES
from openai import OpenAI
from cu2til.prompt.cuda2triton import simple_initial_prompt
import dotenv
from cu2til.trans.dev_.llm import get_api_param_openai_default
from llm import openai_llm_call, CallingIdentifier, get_api_params_method, make_openai_message_system, make_openai_message_user

# Load environment variables
dotenv.load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run pass@n testing for CUDA to Triton translation')
    parser.add_argument('--model', choices=['qwen', 'gpt', 'gemini', 'deepseek', 'claude'], 
                       default='qwen', help='Model to use for testing')
    parser.add_argument('--max-times', type=int, default=5, 
                       help='Maximum number of attempts (n in pass@n)')
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
    return parser.parse_args()

# Parse arguments
args = parse_args()
run_model = args.model

get_api_param = get_api_param_openai_default
model_name = "DEFAULT_MODEL_NAME"

if run_model == "qwen":
    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    client = OpenAI(
        base_url='http://10.156.112.253:8000/v1',
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
    get_api_param = get_api_params_method(CallingIdentifier.ANTHROPIC_OPENROUTER)
else:
    raise ValueError(f"Unsupported model: {run_model}")

# Constants
DIR_CUDA_ = Path("cuda_")
DIR_TORCH_ = Path("torch_")
DIR_TRITON_ = Path("triton_")
TESTSET_ROOT_DIR = Path("/workspace/cu2til/cases/xpiler")
WORK_DIR = Path(__file__).parent.absolute()
TEMPERATURE = 0.35
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MAX_TIMES = args.max_times  # Number of independent attempts from args
CONSOLE_OUTPUT = args.console and not args.no_console  # Whether to output to console

# Setup work directory
model_name_clean = "".join(c if c.isalnum() else "_" for c in model_name.split("/")[-1].lower())
WORK_DIR = WORK_DIR / f"{model_name_clean}_pass_at_{MAX_TIMES}" / TIMESTAMP
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = WORK_DIR / f"{model_name_clean}_pass_{MAX_TIMES}.log"
logger = logging.getLogger('llm_trans_pass_at_n')
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

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
for handler in handlers:
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Configure available cases based on command line arguments
def get_available_cases():
    """Get available cases based on command line arguments."""
    if args.first_only:
        cases = get_first_case_from_each_type()
    else:
        cases = XPILER_ALL_CASES.copy()
    
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

def get_first_case_from_each_type():
    """Extract the first case from each case type in XPILER_ALL_CASES."""
    first_cases = {}
    for case_type, cases in XPILER_ALL_CASES.items():
        if cases:
            first_cases[case_type] = cases[0]
    return first_cases

def get_last_code_block(resp_content):
    """Extract the last code block from response content."""
    import re
    resp_content = re.sub(r'<thought>.*?</thought>', '', resp_content, flags=re.DOTALL)

    # Find all python code blocks
    python_starts = []
    for match in re.finditer(r'```python\b', resp_content, re.IGNORECASE):
        python_starts.append(match.end())

    if python_starts:
        last_python_start = python_starts[-1]
        remaining_content = resp_content[last_python_start:]
        
        end_match = re.search(r'```', remaining_content)
        if end_match:
            extracted_code = remaining_content[:end_match.start()].strip()
        else:
            extracted_code = remaining_content.strip()
        
        return extracted_code

    # Fallback to any code block
    code_block_pattern = r'```(\w+)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, resp_content, re.DOTALL)
    
    if matches:
        extracted_code = matches[-1][1].strip()
    else:
        extracted_code = resp_content
        logger.warning("Cannot find code block, using original content")
    
    return extracted_code

def run_test(test_work_dir, attempt_num):
    """Run a single test attempt and return success status."""
    logger.debug(f"Running test attempt {attempt_num}...")
    
    # Create logs directory
    logs_dir = test_work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the test and capture output
    log_file = logs_dir / f"triton_test_attempt_{attempt_num}.log"
    cmd = [sys.executable, "check_triton.py"]
    if args.no_perf:
        cmd.append("--no-perf")
    
    try:
        original_cwd = os.getcwd()
        os.chdir(test_work_dir)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        os.chdir(original_cwd)
        
        # Write output to log file
        with open(log_file, 'w') as f:
            f.write(f"=== Test Attempt {attempt_num} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== END OF LOG ===\n")
        
        # Check if test passed
        success = result.returncode == 0 and ("PASSED" in result.stdout)
        
        logger.debug(f"Test attempt {attempt_num} result: {'PASSED' if success else 'FAILED'}")
        return success, log_file
        
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        error_msg = f"Test attempt {attempt_num} timed out after 2 minutes"
        with open(log_file, 'w') as f:
            f.write(f"=== Test Attempt {attempt_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        logger.error(error_msg)
        return False, log_file
    except Exception as e:
        os.chdir(original_cwd)
        error_msg = f"Test attempt {attempt_num} failed with exception: {str(e)}"
        with open(log_file, 'w') as f:
            f.write(f"=== Test Attempt {attempt_num} ===\n")
            f.write(f"ERROR: {error_msg}\n")
        logger.error(error_msg)
        return False, log_file

def generate_triton_code(cuda_code, attempt_num):
    """Generate Triton code using LLM for a single attempt."""
    logger.debug(f"Generating Triton code for attempt {attempt_num}...")
    
    # Create fresh conversation for each attempt (no history)
    conversation_history = [
        make_openai_message_system(
            'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming. You help convert CUDA kernels to Triton kernels while maintaining correctness and performance.'),
        make_openai_message_user(simple_initial_prompt.format(cuda_code=cuda_code))
    ]
    
    try:
        api_params = get_api_param(conversation_history, model_name)
        resp_content = openai_llm_call(client, api_params)
        triton_code = get_last_code_block(resp_content)
        
        logger.debug(f"Successfully generated Triton code for attempt {attempt_num}")
        return triton_code, resp_content
    except Exception as e:
        logger.error(f"Failed to generate Triton code for attempt {attempt_num}: {str(e)}")
        return None, None

def run_pass_at_n_case(case_type, case_name):
    """Run pass@n testing for a single case."""
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Testing case type: {case_type}")
    logger.info(f"📁 Case name: {case_name}")
    logger.info(f"🔢 Max attempts: {MAX_TIMES}")
    logger.info(f"{'='*60}")
    
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
            logger.debug(f"Copy {file_path} to {test_work_dir}")
        else:
            logger.warning(f"Source file {src_file} does not exist")
    
    # Read CUDA code
    cuda_file_path = test_work_dir / DIR_CUDA_ / "kernel.cu"
    if not cuda_file_path.exists():
        logger.error(f"CUDA file not found: {cuda_file_path}")
        return False, None, []
    
    with open(cuda_file_path, "r") as f:
        cuda_code = f.read()
    
    # Create triton directory
    TRITON_DIR = test_work_dir / DIR_TRITON_
    TRITON_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track all attempts
    attempts_log = []
    
    # Run MAX_TIMES independent attempts
    for attempt in range(1, MAX_TIMES + 1):
        logger.info(f"🔄 Starting attempt {attempt}/{MAX_TIMES}...")
        
        # Generate Triton code (independent attempt)
        triton_code, llm_response = generate_triton_code(cuda_code, attempt)
        
        if triton_code is None:
            logger.error(f"Failed to generate code for attempt {attempt}")
            attempts_log.append({
                "attempt": attempt,
                "success": False,
                "error": "Code generation failed"
            })
            continue
        
        # Save generated code
        kernel_path = TRITON_DIR / f"kernel_attempt_{attempt}.py"
        with open(kernel_path, "w") as f:
            f.write(triton_code)
        
        # Copy to kernel.py for testing
        shutil.copy(kernel_path, TRITON_DIR / "kernel.py")
        
        # Save LLM response
        response_file = test_work_dir / "logs" / f"llm_response_attempt_{attempt}.txt"
        response_file.parent.mkdir(parents=True, exist_ok=True)
        with open(response_file, "w") as f:
            f.write(llm_response)
        
        # Test the generated code
        success, log_file = run_test(test_work_dir, attempt)
        
        attempts_log.append({
            "attempt": attempt,
            "success": success,
            "log_file": str(log_file),
            "kernel_file": str(kernel_path)
        })
        
        if success:
            logger.info(f"✅ SUCCESS at attempt {attempt}! (pass@{attempt})")
            return True, attempt, attempts_log
        else:
            logger.info(f"❌ Attempt {attempt} failed")
    
    logger.info(f"❌ All {MAX_TIMES} attempts failed")
    return False, None, attempts_log

def test_cases():
    """Test cases using pass@n methodology."""
    logger.info(f"🚀 Starting pass@{MAX_TIMES} testing...")
    logger.info(f"🤖 Using model: {model_name}")
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   - Max attempts: {MAX_TIMES}")
    logger.info(f"   - Console output: {CONSOLE_OUTPUT}")
    logger.info(f"   - First only: {args.first_only}")
    logger.info(f"   - Skip performance: {args.no_perf}")
    logger.info(f"   - Specific case types: {args.case_types if args.case_types else 'All'}")
    logger.info(f"📋 Available case types: {list(AVAILABLE_CASES.keys())}")
    logger.info(f"📁 Work directory: {WORK_DIR}")
    
    detailed_results = {}
    all_case_results = []
    
    for case_type, case_name in AVAILABLE_CASES.items():
        if not (isinstance(case_name, tuple) or isinstance(case_name, list)):
            case_name = [case_name]
        
        detailed_results[case_type] = []
        
        for case_name_single in case_name:
            try:
                logger.info(f"\n{'🔹'*20} Starting {case_type}/{case_name_single} {'🔹'*20}")
                success, pass_at_n, attempts_log = run_pass_at_n_case(case_type, case_name_single)
                
                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": success,
                    "pass_at_n": pass_at_n,
                    "attempts_log": attempts_log
                }
                detailed_results[case_type].append(case_result)
                all_case_results.append(case_result)
                
                if success:
                    status = f"✅ SUCCESS (pass@{pass_at_n})"
                else:
                    status = f"❌ FAILED (all {MAX_TIMES} attempts)"
                logger.info(f"{'🔹'*15} {case_type}/{case_name_single}: {status} {'🔹'*15}")
                
            except Exception as e:
                logger.error(f"Error in {case_type}/{case_name_single}: {e}")
                case_result = {
                    "case_type": case_type,
                    "case_name": case_name_single,
                    "success": False,
                    "pass_at_n": None,
                    "error": str(e),
                    "attempts_log": []
                }
                detailed_results[case_type].append(case_result)
                all_case_results.append(case_result)
    
    # Save detailed results
    results_file = WORK_DIR / f"results_{model_name_clean}_pass_{MAX_TIMES}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "max_times": MAX_TIMES,
            "timestamp": TIMESTAMP,
            "detailed_results": detailed_results,
            "all_case_results": all_case_results
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 PASS@{MAX_TIMES} TESTING SUMMARY")
    logger.info(f"{'='*80}")
    
    total_success = 0
    total_cases = 0
    pass_at_n_distribution = {}
    
    for case_type, case_results in detailed_results.items():
        case_type_success = sum(1 for result in case_results if result["success"])
        case_type_total = len(case_results)
        total_success += case_type_success
        total_cases += case_type_total
        
        case_type_status = "✅" if case_type_success == case_type_total else "❌" if case_type_success == 0 else "⚠️ "
        logger.info(f"{case_type_status} {case_type:<15} ({case_type_success}/{case_type_total})")
        
        for result in case_results:
            if result["success"]:
                pass_n = result["pass_at_n"]
                pass_at_n_distribution[pass_n] = pass_at_n_distribution.get(pass_n, 0) + 1
                status = f"  ✅ pass@{pass_n}"
            else:
                status = f"  ❌ failed"
            
            error_info = f" - {result.get('error', '')}" if not result["success"] and 'error' in result else ""
            logger.info(f"{status} {result['case_name']}{error_info}")
        logger.info("")
    
    # Pass@n distribution
    logger.info(f"📊 Pass@n distribution:")
    for n in sorted(pass_at_n_distribution.keys()):
        count = pass_at_n_distribution[n]
        logger.info(f"   pass@{n}: {count} cases")
    
    # Calculate pass@k rates
    successful_cases = [result for result in all_case_results if result["success"]]
    logger.info(f"\n📈 Pass@k rates:")
    for k in range(1, MAX_TIMES + 1):
        pass_at_k = sum(1 for result in successful_cases if result["pass_at_n"] <= k)
        rate = pass_at_k / total_cases * 100 if total_cases > 0 else 0
        logger.info(f"   pass@{k}: {pass_at_k}/{total_cases} ({rate:.1f}%)")
    
    logger.info(f"{'='*80}")
    logger.info(f"🏆 OVERALL TOTAL: {total_success}/{total_cases} cases succeeded")
    logger.info(f"📊 Case type success rate: {sum(1 for case_type, results in detailed_results.items() if all(r['success'] for r in results))}/{len(detailed_results)} case types fully passed")
    logger.info(f"📁 Results saved to: {results_file}")
    logger.info(f"📋 Log file: {log_file}")
    logger.info(f"{'='*80}")
    
    return {"detailed_results": detailed_results, "all_case_results": all_case_results}

if __name__ == "__main__":
    """
    Examples of usage:
    
    # Test all cases with pass@5 using qwen model
    python llm_trans_pass_at_n.py
    
    # Test with pass@10 using claude model
    python llm_trans_pass_at_n.py --model claude --max-times 10
    
    # Test only specific case types
    python llm_trans_pass_at_n.py --case-types conv2d gemm layernorm
    
    # Test only first case from each type with no console output
    python llm_trans_pass_at_n.py --first-only --no-console
    
    # Test with different model and max attempts, skip performance testing
    python llm_trans_pass_at_n.py --model gpt --max-times 3 --case-types softmax relu --no-perf
    """
    
    # Validate configuration
    if not AVAILABLE_CASES:
        logger.error("No cases available for testing. Check your case types filter.")
        sys.exit(1)
    
    # Test all cases using pass@n methodology
    test_cases()

# python /workspace/cu2til/trans/dev_/llm_trans_pass_at_n.py --model qwen --max-times 10 --no-perf