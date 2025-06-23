import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['TORCH_CUDA_ARCH_LIST'] = "Ada"
import torch
import importlib.util
from pathlib import Path
import time
import subprocess
import tempfile
import sys
import logging
from datetime import datetime

def compile_cuda_kernel_from_file(kernel_path: str, verbose: bool = False) -> callable:
    """编译CUDA内核"""
    spec = importlib.util.spec_from_file_location(
        Path(kernel_path).stem,
        kernel_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'forward')

def compile_cuda_extension(cuda_file: str, build_dir: str = './build', name: str = "cuda_kernel"):
    """编译CUDA扩展"""
    from torch.utils.cpp_extension import load
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    try:
        extension = load(
            name=name,
            sources=[cuda_file],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_80,code=sm_80', '-gencode=arch=compute_90,code=sm_90'],
            verbose=True,
            build_directory=build_dir
        )
        return extension
    except Exception as e:
        return str(e)

def load_torch_ref(ref_path: str):
    """加载PyTorch参考实现"""
    spec = importlib.util.spec_from_file_location("torch_ref", ref_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def benchmark_kernel(kernel_func, inputs):
    from tools.performer import KernelPerfBench
    return KernelPerfBench.func_perf_test_median(kernel_func, inputs)

def eval_triton_kernel(base_dir: str, model_name: str = "new_test", logger=logging.getLogger(__name__), time_str: str = None, logfile_prefix: str = "", return_result: bool = False, timestamp_log_dir: str = None):
        
    # 文件路径
    # 动态确定triton文件路径
    if timestamp_log_dir and logfile_prefix.startswith("round"):
        # 多轮生成中的测试，使用对应轮次的文件
        round_num = logfile_prefix.replace("round", "").replace("_", "")
        triton_file = f"{timestamp_log_dir}/triton_round{round_num}.py"
    elif timestamp_log_dir:
        # 使用时间戳目录中的triton.py（最终版本）
        triton_file = f"{timestamp_log_dir}/triton.py"
        
    cuda_file = f"{base_dir}/cuda_ref.cu"  
    torch_ref_file = f"{base_dir}/torch_ref.py"
    log_dir = f"{base_dir}/logs"
    
    # 根据模型名创建子文件夹
    model_name_clean = model_name.replace('.', '_').replace('/', '_').replace('-', '_')
    model_log_dir = f"{log_dir}/{model_name_clean}"
    
    if time_str is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 如果提供了时间戳文件夹，将eval日志保存到其中，否则保存到模型文件夹
    if timestamp_log_dir is not None:
        log_file = f"{timestamp_log_dir}/{logfile_prefix}eval.log"
        # 只为最终的eval（不是round级别的）在外面保留最新副本
        if not logfile_prefix.startswith("round"):
            latest_eval_file = f"{model_log_dir}/eval_latest.log"
        else:
            latest_eval_file = None
    else:
        os.makedirs(model_log_dir, exist_ok=True)
        log_file = f"{model_log_dir}/{logfile_prefix}eval_{time_str}.log"
        latest_eval_file = None
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if latest_eval_file:
        os.makedirs(os.path.dirname(latest_eval_file), exist_ok=True)
    
    # 设置日志处理器，如果是多轮对话，使用追加模式
    # 清除之前的处理器
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    
    # 添加主日志处理器（时间戳文件夹中的日志）
    main_handler = logging.FileHandler(log_file, mode='a')
    logger.addHandler(main_handler)
    
    # 如果需要，添加最新日志副本处理器
    if latest_eval_file:
        latest_handler = logging.FileHandler(latest_eval_file, mode='w')  # 最新日志使用覆盖模式
        logger.addHandler(latest_handler)
    
    logger.setLevel(logging.DEBUG)
    
    # 添加轮次分隔符
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    round_info = ""
    if logfile_prefix.startswith("round"):
        round_info = f" - {logfile_prefix.replace('_', '').upper()}"
    logger.info(f"\n{'='*60}")
    logger.info(f"测试开始时间: {current_time}{round_info}")
    logger.info(f"{'='*60}")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.info("❌ CUDA不可用，跳过测试")
        if return_result:
            return {"success": False, "error": "CUDA不可用"}
        return
    
    # 加载PyTorch参考实现获取测试用例
    logger.info("📁 加载测试用例...")
    torch_ref = load_torch_ref(torch_ref_file)
    test_inputs = torch_ref.get_inputs()
    
    # 将输入移到GPU
    cuda_inputs = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in test_inputs]
    print(f"📊 测试矩阵大小: {cuda_inputs[0].shape}")
    
    # 加载Triton内核
    logger.info(f"⚡ 加载Triton内核: {triton_file}")
    try:
        triton_func = compile_cuda_kernel_from_file(triton_file, verbose=False)
        logger.info("✅ Triton内核加载成功")
    except Exception as e:
        logger.info(f"❌ Triton内核加载失败: {e}")
        if return_result:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        return
    
    # 编译CUDA内核
    logger.info("🔧 编译CUDA内核...")
    try:
        cuda_extension = compile_cuda_extension(cuda_file, build_dir=f"{base_dir}/build")
        if cuda_extension is None:
            logger.info("❌ CUDA内核编译失败")
            if return_result:
                return {"success": False, "error": "CUDA内核编译失败"}
            return
        cuda_func = getattr(cuda_extension, 'forward')
        logger.info("✅ CUDA内核编译成功")
    except Exception as e:
        logger.info(f"❌ CUDA内核编译失败: {e}")
        if return_result:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        return
    
    # 功能测试
    logger.info("\n🧪 开始功能正确性测试...")
    
    try:
        triton_result = triton_func(*cuda_inputs)
        cuda_result = cuda_func(*cuda_inputs)
        torch_result = torch_ref.module_fn(*cuda_inputs)
        atol = 1e-1
        rtol = 1e-1
        # 比较结果
        triton_cuda_match = torch.allclose(triton_result, cuda_result, atol=atol, rtol=rtol)
        triton_torch_match = torch.allclose(triton_result, torch_result, atol=atol, rtol=rtol)
        cuda_torch_match = torch.allclose(cuda_result, torch_result, atol=atol, rtol=rtol)
        
        logger.info(f"🔍 Triton vs CUDA 匹配: {'✅' if triton_cuda_match else '❌'}")
        logger.info(f"🔍 Triton vs PyTorch 匹配: {'✅' if triton_torch_match else '❌'}")
        logger.info(f"🔍 CUDA vs PyTorch 匹配: {'✅' if cuda_torch_match else '❌'}")
        
        if not triton_cuda_match:
            max_diff = torch.max(torch.abs(triton_result - cuda_result)).item()
            logger.info(f"   最大差异: {max_diff:.2e}")
        
        if not (triton_cuda_match and cuda_torch_match):
            logger.info("❌ 功能测试失败，跳过性能测试")
            if return_result:
                return {"success": False, "error": "功能测试失败：结果不匹配"}
            return
            
    except Exception as e:
        logger.info(f"❌ 功能测试出错: {e}")
        if return_result:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        return
    
    # 性能测试
    logger.info("\n⏱️  开始性能测试...")
    triton_time = benchmark_kernel(triton_func, cuda_inputs)
    
    logger.info("测试CUDA内核性能...")
    cuda_time = benchmark_kernel(cuda_func, cuda_inputs)
    
    logger.info("测试PyTorch参考性能...")
    torch_time = benchmark_kernel(torch_ref.module_fn, cuda_inputs)
    
    # 性能结果
    logger.info(f"\n📈 性能测试结果:")
    logger.info(f"   Triton:   {triton_time:.3f} ms")
    logger.info(f"   CUDA:     {cuda_time:.3f} ms")
    logger.info(f"   PyTorch:  {torch_time:.3f} ms")
    
    # 加速比
    if triton_time > 0:
        triton_cuda_speedup = cuda_time / triton_time
        triton_torch_speedup = torch_time / triton_time
        logger.info(f"\n🚀 加速比:")
        logger.info(f"   Triton vs CUDA: {triton_cuda_speedup:.2f}x")
        logger.info(f"   Triton vs PyTorch: {triton_torch_speedup:.2f}x")

    logger.info("\n✅ 测试完成!")
    logger.info(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}\n")
    
    if return_result:
        return {
            "success": True,
            "performance": {
                "triton_time": triton_time,
                "cuda_time": cuda_time,
                "torch_time": torch_time,
                "triton_cuda_speedup": triton_cuda_speedup,
                "triton_torch_speedup": triton_torch_speedup
            },
            "correctness": {
                "triton_cuda_match": triton_cuda_match,
                "triton_torch_match": triton_torch_match,
                "cuda_torch_match": cuda_torch_match
            }
        }
