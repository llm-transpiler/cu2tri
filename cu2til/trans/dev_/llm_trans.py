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
DIR_CUDA_ = Path("cuda_")
DIR_TORCH_ = Path("torch_")
DIR_TRITON_ = Path("triton_")
TESTSET_ROOT_DIR = Path("/workspace/cu2til/cases/xpiler_subset")
WORK_DIR = Path(__file__).parent.absolute()
TEMPERATURE = 0.35
model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
model_name = "gemini-2.5-pro"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
WORK_DIR = Path(__file__).parent.absolute() / "".join(c if c.isalnum() else "_" for c in model_name.split("/")[-1].lower()) / TIMESTAMP
# 执行自动化测试和修复
max_rounds = 5
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if "qwen" in model_name.lower():
    print(f"Using Qwen model: {model_name}")
    client = OpenAI(
        base_url='http://10.156.112.253:8000/v1',  # api_base
        api_key="EMPTY"
    )
elif "gemini" in model_name.lower():
    print(f"Using Gemini model: {model_name}")
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_api_key
    )
else:
    raise ValueError(f"Unsupported model: {model_name}")
# Import case configuration
from xpiler_case_config import XPILER_ALL_CASES

# Get the first case from each case type for testing
def get_first_case_from_each_type():
    """Extract the first case from each case type in XPILER_ALL_CASES."""
    first_cases = {}
    for case_type, cases in XPILER_ALL_CASES.items():
        if cases:  # Make sure the list is not empty
            first_cases[case_type] = cases[0]
    return first_cases

# Configuration - can be changed to test different cases
AVAILABLE_FIRST_CASES = get_first_case_from_each_type()

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
    files_to_copy = [DIR_TORCH_ / "ref.py", DIR_CUDA_ / "kernel.cu", "check_cuda.py", "check_triton.py", "get_data.py"]
    
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
        {
            'role': 'system', 
            'content': 'You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming. You help convert CUDA kernels to Triton kernels while maintaining correctness and performance.'
        },
        {
            'role': 'user', 
            'content': simple_initial_prompt.format(cuda_code=cuda_code)
        }
    ]
    
    # Configure API call based on model type
    api_params = {
        "messages": conversation_history,
        "model": model_name,
        "max_tokens": 65536,
        "stream": True,
        "temperature": TEMPERATURE
    }
    
    # Add thinking config for Gemini models
    if "gemini" in model_name.lower():
        api_params["extra_body"] = {
            'extra_body': {
                "google": {
                "thinking_config": {
                    "thinking_budget": -1,
                    "include_thoughts": True
                }
                }
            }
        }
    
    completion = client.chat.completions.create(**api_params)
    
    resp_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            resp_content += content
    
    resp_content = resp_content.strip()
    
    # 将LLM的回复添加到对话历史中
    conversation_history.append({'role': 'assistant', 'content': resp_content})

    
    TRITON_DIR = test_work_dir / DIR_TRITON_
    TRITON_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRITON_DIR / "kernel.py", "w") as f:
        f.write(get_last_code_block(resp_content))
    
    print(f"✅ Triton code generated successfully")
    
    # Save initial conversation history (round 1)
    save_conversation_history(conversation_history, test_work_dir, round_num=1, timestamp=TIMESTAMP)

    # 开始自动化测试和修复流程
    print(f"\n🔄 Starting automated testing and fixing process...")
    
    return run_testing_loop(conversation_history, test_work_dir)

def get_last_code_block(resp_content):
    """Extract the last code block from response content, preferring the last ```python ... ``` block if present.
    
    Handles cases where ```python tags may not be properly closed.
    """
    import re
    resp_content = re.sub(r'<thought>.*?</thought>', '', resp_content, flags=re.DOTALL)
    
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
            conversation_file = logs_dir / f"conversation_history_round_{round_num}.json"
        else:
            conversation_file = logs_dir / f"conversation_history_{timestamp}.json"
        
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Conversation history saved to {conversation_file}")
        return str(conversation_file)
    except Exception as e:
        print(f"⚠️ Failed to save conversation history: {e}")
        return None

def save_conversation_summary(conversation_history, test_work_dir, final_status, timestamp=TIMESTAMP):
    """Save a human-readable summary of the conversation."""
    try:
        logs_dir = test_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = logs_dir / f"conversation_summary_{timestamp}.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# CUDA to Triton Conversion Conversation Summary\n\n")
            f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model**: {model_name}\n")
            f.write(f"**Final Status**: {final_status}\n")
            f.write(f"**Total Messages**: {len(conversation_history)}\n\n")
            
            for i, msg in enumerate(conversation_history):
                role = msg['role'].upper()
                content = msg['content']
                
                f.write(f"## Message {i+1}: {role}\n\n")
                if role == 'SYSTEM':
                    f.write(f"```\n{content}\n```\n\n")
                elif role == 'USER':
                    # Truncate very long user messages for readability
                    if len(content) > 1000:
                        f.write(f"{content[:500]}\n...(truncated)...\n{content[-500:]}\n\n")
                    else:
                        f.write(f"{content}\n\n")
                else:  # ASSISTANT
                    # Truncate very long assistant messages
                    if len(content) > 2000:
                        f.write(f"{content[:1000]}\n...(truncated)...\n{content[-1000:]}\n\n")
                    else:
                        f.write(f"{content}\n\n")
                
                f.write("---\n\n")
        
        print(f"📄 Conversation summary saved to {summary_file}")
        return str(summary_file)
    except Exception as e:
        print(f"⚠️ Failed to save conversation summary: {e}")
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
    cmd = [sys.executable, "check_triton.py"]#, "--no-perf"]
    
    try:
        # Change to target directory for test execution
        original_cwd = os.getcwd()
        os.chdir(test_work_dir)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
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
        conversation_history.append({
            "role": "user", 
            "content": prompt
        })
        
        # Configure API call based on model type
        api_params = {
            "messages": conversation_history,
            "model": model_name,
            "max_tokens": 65536,
            "stream": True,
            "temperature": TEMPERATURE
        }
        
        # Add thinking config for Gemini models
        if "gemini" in model_name.lower():
            api_params["extra_body"] = {
                'extra_body': {
                    "google": {
                    "thinking_config": {
                        "thinking_budget": -1,
                        "include_thoughts": True
                    }
                    }
                }
            }
        
        completion = client.chat.completions.create(**api_params)
        
        resp_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                resp_content += content
        
        fixed_code = resp_content.strip()
        
        # Add LLM response to conversation history
        conversation_history.append({"role": "assistant", "content": resp_content})
        
        fixed_code = get_last_code_block(fixed_code)
        return fixed_code
        
    except Exception as e:
        print(f"❌ Failed to get LLM feedback: {str(e)}")
        return None

def run_testing_loop(conversation_history, test_work_dir):
    """Run the testing and fixing loop."""
    for round_num in range(1, max_rounds + 1):
        success, stdout, stderr, log_file = run_test_round(round_num, test_work_dir)
        
        if success:
            print(f"✅ Test round {round_num} PASSED! Triton kernel is working correctly.")
            print(f"📊 Final results saved in {log_file}")
            
            # 成功时只保存总结，对话历史已经在之前保存过了
            save_conversation_summary(conversation_history, test_work_dir, "SUCCESS", timestamp=TIMESTAMP)
            return True
        else:
            print(f"❌ Test round {round_num} FAILED.")
            print(f"📋 Error details saved in {log_file}")
            
            if round_num < max_rounds:
                print(f"🔧 Attempting to fix with LLM feedback...")
                
                # Get feedback from LLM using cumulative conversation history
                # 这将产生第(round_num+1)轮对话
                fixed_code = get_feedback_from_llm(round_num, stdout, stderr, test_work_dir, conversation_history)
                
                if fixed_code:
                    # Save the fixed code
                    kernel_path = test_work_dir / DIR_TRITON_ / "kernel.py"
                    with open(kernel_path, 'w') as f:
                        f.write(fixed_code)
                    print(f"💾 Updated kernel.py with LLM feedback")
                    
                    # 保存第(round_num+1)轮对话历史
                    save_conversation_history(conversation_history, test_work_dir, round_num + 1, timestamp=TIMESTAMP)
                else:
                    print(f"❌ Failed to get valid LLM feedback for round {round_num}")
                    save_conversation_summary(conversation_history, test_work_dir, "FAILED - No valid feedback", timestamp=TIMESTAMP)
                    break
            else:
                print(f"❌ Maximum rounds ({max_rounds}) reached. Manual intervention required.")
                save_conversation_summary(conversation_history, test_work_dir, "FAILED - Max rounds reached", timestamp=TIMESTAMP)
                break
    
    # Final save if loop completed without break due to other reasons
    else:
        save_conversation_summary(conversation_history, test_work_dir, "COMPLETED - Loop end", timestamp=TIMESTAMP)
    
    return False

def test_cases():
    """Test the first case from each case type."""
    print(f"🚀 Starting batch testing of first cases from each type...")
    print(f"📋 Available case types: {list(AVAILABLE_FIRST_CASES.keys())}")
    
    results = {}
    for case_type, case_name in AVAILABLE_FIRST_CASES.items():
        try:
            print(f"\n{'🔹'*20} Starting {case_type} {'🔹'*20}")
            success = run_single_case_translation(case_type, case_name)
            results[case_type] = {"case_name": case_name, "success": success}
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{'🔹'*15} {case_type}: {status} {'🔹'*15}")
        except Exception as e:
            print(f"❌ Error in {case_type}: {e}")
            results[case_type] = {"case_name": case_name, "success": False, "error": str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH TESTING SUMMARY")
    print(f"{'='*60}")
    
    success_count = 0
    for case_type, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {case_type:<15} - {result['case_name']}")
        if result["success"]:
            success_count += 1
    
    print(f"{'='*60}")
    print(f"🏆 Total: {success_count}/{len(results)} cases succeeded")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    # Configuration options:
    
    # # Option 1: Test a single specific case
    # case_type = "conv2d"  # Change this to test different case types
    # if case_type in AVAILABLE_FIRST_CASES:
    #     case_name = AVAILABLE_FIRST_CASES[case_type]
    #     print(f"📋 Available case types: {list(AVAILABLE_FIRST_CASES.keys())}")
    #     run_single_case_translation(case_type, case_name)
    # else:
    #     print(f"❌ Case type '{case_type}' not found in available cases")
    #     print(f"📋 Available case types: {list(AVAILABLE_FIRST_CASES.keys())}")
    
    # # Option 2: Test all first cases (uncomment to enable)
    test_cases()
