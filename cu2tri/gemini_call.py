import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import dotenv
import importlib.util
from pathlib import Path
import re
from datetime import datetime
import time

from google import genai
from google.genai import types

dotenv.load_dotenv()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

GEMINI_MODEL_NAME = 'gemini-2.0-flash'  # Or 'gemini-pro', 'gemini-1.0-pro' etc.
GEMINI_MODEL_NAME = 'gemini-2.5-pro'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Fallback for local development if not set in environment
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MAX_RETRIES = 5
BASE_SLEEP_TIME = 3

def compile_cuda_kernel_from_file(kernel_path: str, verbose: bool) -> callable:
    """Compile CUDA kernel from file"""
    spec = importlib.util.spec_from_file_location(
        Path(kernel_path).stem,
        kernel_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'forward')

def generate_with_gemini(contents_list):
    """
    使用Gemini生成内容，支持多轮对话
    contents_list: 包含types.Content对象的列表
    """
    for _ in range(MAX_RETRIES):
        try:
            # response = client.models.generate_content(
            #     model=GEMINI_MODEL_NAME,
            #     contents=contents_list,
            #     config=types.GenerateContentConfig(
            #         system_instruction='You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming.',
            #         max_output_tokens=8192 * 2,
            #         temperature=0.3,
            #         thinking_config = types.ThinkingConfig(
            #             thinking_budget=-1,
            #         ),
            #         response_mime_type="text/plain",
            #     ))
            # return response.text
            # 使用 generate_content_stream 获取流式响应
            res_list = []
            thoughts = ""
            answer = ""
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL_NAME,
                contents=contents_list,
                config=types.GenerateContentConfig(
                    system_instruction='You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming.',
                    max_output_tokens=8192 * 2,
                    temperature=0.3,
                    # thinking_config = types.ThinkingConfig(
                    #     thinking_budget=-1,
                    # ),
                    thinking_config=types.ThinkingConfig(thinking_budget=-1, 
                                                         include_thoughts=True),
                    response_mime_type="text/plain",
                )
            ):
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if part.thought:
                        if not thoughts:
                            print("Thoughts summary")
                        # print(part.text)
                        thoughts += part.text
                    else:
                        if not answer:
                            print("Answer:")
                        # print(part.text)
                        answer += part.text
            # 组装最终的完整响应
            final_response = "<thoughts>" + thoughts + "</thoughts>" + "<answer>" + answer + "</answer>"
            return final_response
        except Exception as e:
            print(f"Error generating with Gemini: {e}")
            time.sleep(BASE_SLEEP_TIME * (2 ** _))
    raise Exception("Failed to generate with Gemini")

def test_triton_kernel(test_dir: str, time_str: str = None, iteration: int = None, timestamp_log_dir: str = None):
    """测试Triton内核的编译和运行正确性"""
    try:
        from eval_triton import eval_triton_kernel
        # 如果提供了迭代信息，添加到日志前缀中
        logfile_prefix = ""
        if iteration is not None:
            logfile_prefix = f"round{iteration}_"
        result = eval_triton_kernel(test_dir, model_name=GEMINI_MODEL_NAME, return_result=True, time_str=time_str, logfile_prefix=logfile_prefix, timestamp_log_dir=timestamp_log_dir)
        return result
    except Exception as e:
        import traceback
        return {
            "success": False, 
            "error": str(e), 
            "traceback": traceback.format_exc()
        }

def extract_code_from_response(response: str) -> str:
    """从响应中提取Python代码"""
    # 首先尝试提取<answer></answer>中的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)
        
        # 在answer内容中尝试提取```python代码块
        code_match = re.search(r'```python\n(.*?)\n```', answer_content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # 在answer内容中尝试提取```代码块
        code_match = re.search(r'```\n(.*?)\n```', answer_content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # 如果answer中没有代码块，返回answer的全部内容
        return answer_content
    
    # 如果没有找到answer标签，使用原有逻辑
    # 尝试提取```python代码块
    code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        return code_match.group(1)
    
    # 尝试提取```代码块
    code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        return code_match.group(1)
    
    # 如果没有找到代码块，返回整个响应
    return response

def generate_triton_kernel_with_feedback(test_dir: str, model_name: str, time_str: str = None, max_iterations: int = 5):
    """通过多轮对话和反馈生成Triton内核代码"""
    cuda_file_path = f"{test_dir}/cuda_ref.cu"
    torch_file_path = f"{test_dir}/torch_ref.py"
    # triton_now_path = f"{test_dir}/triton_new.py"
    log_dir = f"{test_dir}/logs"
    
    # 根据模型名创建子文件夹
    model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
    model_log_dir = f"{log_dir}/{model_name_clean}"
    
    if time_str is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建时间戳文件夹，文件名不再包含时间戳
    timestamp_log_dir = f"{model_log_dir}/{time_str}"
    triton_st_path = f"{timestamp_log_dir}/triton.py"
    conversation_log_path = f"{timestamp_log_dir}/conversation.log"
    
    # 为eval创建统一的时间戳，确保多轮对话使用同一个eval日志文件
    eval_time_str = time_str
    
    os.makedirs(timestamp_log_dir, exist_ok=True)
    
    with open(cuda_file_path, "r") as f:
        cuda_code = f.read()
    
    # 初始化对话历史
    conversation = []
    
    # 第一轮：初始生成请求
    import prompt
    initial_prompt = prompt.simple_initial_prompt.format(cuda_code=cuda_code)

    conversation.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=initial_prompt)]
    ))
    
    # 初始化对话日志
    def write_conversation_log(conversation_list, iteration_info=""):
        """将对话内容写入日志文件"""
        with open(conversation_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== CUDA到Triton代码转换对话日志 ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试目录: {test_dir}\n")
            f.write(f"最大迭代次数: {max_iterations}\n")
            if iteration_info:
                f.write(f"{iteration_info}\n")
            f.write(f"="*60 + "\n\n")
            
            for i, content in enumerate(conversation_list):
                f.write(f"{'='*20} 第{(i//2)+1}轮 - {content.role.upper()} {'='*20}\n")
                for part in content.parts:
                    if hasattr(part, 'text'):
                        f.write(f"{part.text}\n")
                f.write(f"\n{'='*60}\n\n")
    
    # 记录初始prompt
    write_conversation_log(conversation, "初始prompt已生成")

    for iteration in range(max_iterations):
        print(f"第 {iteration + 1} 轮生成...")
        
        # 生成代码
        response = generate_with_gemini(conversation)
        
        # 添加模型回复到对话历史
        conversation.append(types.Content(
            role="model", 
            parts=[types.Part.from_text(text=response)]
        ))
        
        # 更新对话日志
        write_conversation_log(conversation, f"第{iteration + 1}轮 - 模型已回复")
        
        # 提取代码
        triton_code = extract_code_from_response(response)
        
        # 保存当前轮次的代码，文件名不包含时间戳
        round_triton_path = f"{timestamp_log_dir}/triton_round{iteration+1}.py"
        with open(round_triton_path, "w") as f:
            f.write(triton_code)
        print(f"第{iteration+1}轮代码已保存到: {round_triton_path}")
        
        # # 写入当前工作文件（供测试使用）
        # with open(triton_now_path, "w") as f:
        #     f.write(triton_code)
        
        # print(f"当前代码已写入工作文件: {triton_now_path}")
        
        # 测试代码
        print("正在测试生成的代码...")
        test_result = test_triton_kernel(test_dir, time_str=eval_time_str, iteration=iteration+1, timestamp_log_dir=timestamp_log_dir)
        
        if test_result.get("success", False):
            print("✅ 代码测试成功！")
            # 保存最终成功版本，文件名不包含时间戳
            final_triton_path = f"{timestamp_log_dir}/triton_final.py"
            with open(final_triton_path, "w") as f:
                f.write(triton_code)
            with open(triton_st_path, "w") as f:
                f.write(triton_code)
            print(f"最终成功代码已保存到: {final_triton_path}")
            print(f"最终代码已保存到: {triton_st_path}")
            
            # 打印性能对比结果
            if "performance" in test_result:
                print(f"性能对比结果:")
                perf_info = test_result["performance"]
                if isinstance(perf_info, dict):
                    for key, value in perf_info.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {perf_info}")
            
            # 记录成功结果到日志
            success_info = f"第{iteration + 1}轮 - 代码测试成功！\n"
            if "performance" in test_result:
                success_info += f"性能结果: {test_result['performance']}"
            write_conversation_log(conversation, success_info)
            
            # 创建代码版本摘要
            create_code_version_summary(timestamp_log_dir, iteration + 1, success=True)
            
            return triton_code
        else:
            print(f"❌ 代码测试失败: {test_result.get('error', '未知错误')}")
            
            if iteration == max_iterations - 1:
                print(f"已达到最大迭代次数 ({max_iterations})，停止尝试")
                break
            
            # 准备反馈信息
            error_info = test_result.get('error', '未知错误')
            traceback_info = test_result.get('traceback', '')
            
            feedback_prompt = prompt.feedback_prompt.format(error_info=error_info, traceback_info=traceback_info)

            conversation.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=feedback_prompt)]
            ))
            
            # 记录反馈信息到日志
            error_info_log = f"第{iteration + 1}轮 - 代码测试失败，已提供错误反馈\n错误: {error_info}"
            write_conversation_log(conversation, error_info_log)
    
    # 如果所有迭代都失败了，仍然保存最后一次的代码
    last_attempt_path = f"{timestamp_log_dir}/triton_last_attempt.py"
    with open(last_attempt_path, "w") as f:
        f.write(triton_code)
    with open(triton_st_path, "w") as f:
        f.write(triton_code)
    print(f"最后尝试的代码已保存到: {last_attempt_path}")
    
    # 记录最终失败日志
    final_log = f"所有{max_iterations}轮迭代都失败了，保存最后一次生成的代码"
    write_conversation_log(conversation, final_log)
    
    # 创建代码版本摘要
    create_code_version_summary(timestamp_log_dir, max_iterations, success=False)
    
    return triton_code

def create_code_version_summary(timestamp_log_dir: str, total_rounds: int, success: bool = True):
    """创建代码版本摘要文件"""
    summary_path = f"{timestamp_log_dir}/code_versions_summary.md"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Triton代码生成版本摘要\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**总轮次**: {total_rounds}\n")
        f.write(f"**最终结果**: {'✅ 成功' if success else '❌ 失败'}\n\n")
        
        f.write("## 代码版本列表\n\n")
        
        # 列出所有轮次的代码文件
        for round_num in range(1, total_rounds + 1):
            round_file = f"triton_round{round_num}.py"
            f.write(f"- **第{round_num}轮**: `{round_file}`\n")
        
        if success:
            f.write(f"- **最终成功版本**: `triton_final.py` ✅\n")
        else:
            f.write(f"- **最后尝试版本**: `triton_last_attempt.py` ❌\n")
        
        f.write(f"\n## 相关文件\n\n")
        f.write(f"- **对话日志**: `conversation.log`\n")
        f.write(f"- **评估日志**: 查看对应的eval日志文件\n")
        f.write(f"- **最终工作版本**: `../triton_new.py`\n")
        f.write(f"- **存档版本**: `triton.py`\n")
    
    print(f"代码版本摘要已创建: {summary_path}")

def generate_triton_kernel(test_dir: str, model_name: str = GEMINI_MODEL_NAME, time_str: str = None):
    """生成Triton内核代码（支持多轮对话反馈）"""
    return generate_triton_kernel_with_feedback(test_dir, model_name, time_str)
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from eval_triton import eval_triton_kernel
    # test_dir = '/workspace/monocases/cu2tri/1_Square_matrix_multiplication_'
    # time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    # generate_triton_kernel(test_dir, model_name=GEMINI_MODEL_NAME, time_str=time_str)
    # # 为独立调用的eval创建时间戳文件夹
    # model_name_clean = GEMINI_MODEL_NAME.replace('.', '_').replace('/', '_').replace('-', '_')
    # timestamp_log_dir = f"{test_dir}/logs/{model_name_clean}/{time_str}"
    # eval_triton_kernel(test_dir, model_name=GEMINI_MODEL_NAME, timestamp_log_dir=timestamp_log_dir)
    # exit(0)
    levels = ['level1']
    for level in levels:
        output_dir = f"outputs/kernelbench_c/{level}"
        all_dirs = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[0]))
        for dir_name in all_dirs:
            test_dir = f"{output_dir}/{dir_name}"
            # os.system(f"rm -rf {test_dir}/logs/*")
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            try:
                print(f"Generating triton kernel for {test_dir}")
                generate_triton_kernel(test_dir, time_str=time_str)
                # 为批量处理的eval创建时间戳文件夹
                model_name_clean = GEMINI_MODEL_NAME.replace('.', '_').replace('/', '_').replace('-', '_')
                timestamp_log_dir = f"{test_dir}/logs/{model_name_clean}/{time_str}"
                eval_triton_kernel(test_dir, model_name=GEMINI_MODEL_NAME, time_str=time_str, timestamp_log_dir=timestamp_log_dir)
            except Exception as e:
                import traceback
                print(f"Error generating triton kernel: {e}", traceback.format_exc())
                continue
