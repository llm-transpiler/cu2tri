"""
Example tasks for GPU Management API

Shows how to define custom tasks that can be submitted to the GPU management system.
"""

import asyncio
import time
import logging
import os
import multiprocessing

logger = logging.getLogger(__name__)


def simple_cpu_task(message: str, duration: int = 5) -> dict:
    """A simple CPU task that doesn't require GPU"""
    logger.info(f"Starting CPU task: {message}")
    time.sleep(duration)
    logger.info(f"Completed CPU task: {message}")
    return {"message": message, "duration": duration, "result": "success"}


async def async_cpu_task(message: str, duration: int = 3) -> dict:
    """An async CPU task"""
    logger.info(f"Starting async CPU task: {message}")
    await asyncio.sleep(duration)
    logger.info(f"Completed async CPU task: {message}")
    return {"message": message, "duration": duration, "result": "success"}


def gpu_compute_task(n: int = 1000000) -> dict:
    """A task that uses GPU for computation"""
    # return {"status": "success", "result": "no result"}
    try:
        def matmul():
       
            import torch # 可以在这里import, 上面一行也能正常运行
            import gc
            
            logger.info(f"Starting GPU compute task with n={n}")
            # return {"status": "success", "result": "no result"}
            # 获取当前正确的GPU设备, 这一段会持续占用4M的显存
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device = f'cuda:{current_device}'
                logger.info(f"Using device: {device} (CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
            else:
                raise RuntimeError("CUDA is not available")
            # return {"status": "success", "result": "no result"}
            # Create random tensors on the correct GPU device
            # 这一段会持续占用一定显存
            a = torch.randn(n, n, device=device, dtype=torch.float32)
            b = torch.randn(n, n, device=device, dtype=torch.float32)
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            # del a, b
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            # return {"status": "success", "result": "no result"}
            # Perform matrix multiplication
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # 确保计算完成
            end_time = time.time()
            
            del a, b, c
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Get result statistics
            result_mean = float(c.mean().cpu())
            result_std = float(c.std().cpu())
            memory_allocated = torch.cuda.memory_allocated(current_device)
            
            result = {
                "matrix_size": n,
                "computation_time": end_time - start_time,
                "result_mean": result_mean,
                "result_std": result_std,
                "gpu_memory_allocated": memory_allocated,
                "device_used": device,
                "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
                "status": "success"
            }
            
            # 彻底清理GPU显存
            del a, b, c  # 显式删除张量
            gc.collect()  # 强制垃圾回收
            torch.cuda.empty_cache()  # 清空缓存
            torch.cuda.synchronize()  # 同步所有CUDA操作
            
            logger.info(f"Completed GPU compute task: {result}")
            logger.info(f"Memory after cleanup: {torch.cuda.memory_allocated(current_device)} bytes")
            return result
        '''
        import torch
        with torch.no_grad(): # 这一行可以消除其他方法都不行的显存持续占用
            matmul()
        '''# 这样不行, 显存持续占用
        
        with torch.no_grad(): # 这一行可以消除其他方法都不行的显存持续占用
            import torch
            matmul()
        
        
    except Exception as e:
        logger.error(f"GPU compute task failed: {e}")
        # 发生异常时也要清理
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        return {"status": "failed", "error": str(e)}


def memory_intensive_task(memory_gb: float = 1.0) -> dict:
    """A task that allocates significant GPU memory"""
    try:
        import torch
        import gc
        
        logger.info(f"Starting memory intensive task with {memory_gb}GB")
        
        # 获取当前正确的GPU设备
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device = f'cuda:{current_device}'
            logger.info(f"Using device: {device} (CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
        else:
            raise RuntimeError("CUDA is not available")
        
        # Calculate tensor size for desired memory usage
        bytes_per_element = 4  # float32
        elements_needed = int(memory_gb * 1024**3 / bytes_per_element)
        tensor_size = int(elements_needed**0.5)  # Square matrix
        
        # Allocate memory on the correct device
        tensor = torch.randn(tensor_size, tensor_size, device=device, dtype=torch.float32)
        
        # Simple computation
        result_tensor = tensor @ tensor.T
        torch.cuda.synchronize()  # 确保计算完成
        
        # Get statistics
        actual_memory_gb = torch.cuda.memory_allocated(current_device) / 1024**3
        computation_result_mean = float(result_tensor.mean().cpu())
        
        result = {
            "requested_memory_gb": memory_gb,
            "actual_tensor_size": [tensor_size, tensor_size],
            "actual_memory_gb": actual_memory_gb,
            "computation_result_mean": computation_result_mean,
            "device_used": device,
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            "status": "success"
        }
        
        # 彻底清理GPU显存
        del tensor, result_tensor  # 显式删除张量
        gc.collect()  # 强制垃圾回收
        torch.cuda.empty_cache()  # 清空缓存
        torch.cuda.synchronize()  # 同步所有CUDA操作
        
        logger.info(f"Completed memory intensive task: {result}")
        logger.info(f"Memory after cleanup: {torch.cuda.memory_allocated(current_device)} bytes")
        return result
        
    except Exception as e:
        logger.error(f"Memory intensive task failed: {e}")
        # 发生异常时也要清理
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        return {"status": "failed", "error": str(e)}


def check_gpu_assignment() -> dict:
    """Check which GPU this task is actually running on using subprocess"""
    try:
        return {"status": "success", "result": "no result"}
        from eval_.common.mprunner import mp_run
        result = mp_run(
            worker_func=check_gpu_assignment_inner,
            args=()
        )
        logger.info(f"GPU assignment check result: {result}")
        return result.result if result.subproc_success else {"status": "failed", "error": result.error}
    except Exception as e:
        logger.error(f"GPU assignment check failed: {e}")
        return {"status": "failed", "error": str(e)}


def check_gpu_assignment_inner(result_queue: multiprocessing.Queue) -> dict:
    """Check which GPU this task is actually running on"""
    return {"status": "success", "result": "no result"}
    try:
        import torch
        # Get CUDA_VISIBLE_DEVICES setting
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}") # 不知道为什么永远好像指向第一个DEVICE, 而且改了代码直接发起请求好像还是用的原来的代码
        
        # Get current device
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_count = torch.cuda.device_count()
            
            # Try to determine GPU type from device name
            gpu_type = "Unknown"
            # if "H100" in device_name:
            #     gpu_type = "H100"
            # elif "L20" in device_name:
            #     gpu_type = "L20"
            # elif "RTX 6000 Ada" in device_name:
            #     gpu_type = "RTX_6000_Ada"
            # elif "A100" in device_name:
            #     gpu_type = "A100"
            # try:
            #     test_tensor1 = torch.tensor([1.0, 2.0], device=current_device)
            #     test_tensor2 = torch.tensor([3.0, 4.0], device=current_device)
            #     for i in range(10000):
            #         result = test_tensor1 + test_tensor2
            #         # print(f"  GPU functionality test: PASSED {i}")
            #     del test_tensor1, test_tensor2, result
            #     torch.cuda.empty_cache()
            # except Exception as test_e:
            #     print(f"  GPU functionality test: FAILED - {str(test_e)}")
            
        else:
            current_device = None
            device_name = "No CUDA available"
            device_count = 0
            gpu_type = "No GPU"
        
        result = {
            "cuda_visible_devices": visible_devices,
            "current_device_id": current_device,
            "device_name": device_name,
            "gpu_type": gpu_type,
            "total_visible_devices": device_count,
            "status": "success"
        }
        props = torch.cuda.get_device_properties(current_device)
        logger.info(f"Compute capability: {props.major}.{props.minor}")
        logger.info(f"device_name: {device_name}")
        logger.info(f"GPU assignment check: {result}")
        result_queue.put(result)
        return result
        
    except Exception as e:
        logger.error(f"GPU assignment check failed: {e}")
        return {"status": "failed", "error": str(e)}

    finally:
        # 确保结果总是被放入队列，即使出现异常
        try:
            result_queue.put(result)
        except Exception as queue_error:
            logger.error(f"Failed to put result in queue: {queue_error}")
            # 如果队列操作失败，至少记录错误
        
        # 尝试清理PyTorch CUDA上下文
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 强制同步以确保所有CUDA操作完成
                torch.cuda.synchronize()
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup CUDA resources: {cleanup_error}")
    
    return result

async def long_running_task(duration_minutes: int = 5) -> dict:
    """A long-running task for testing queue management"""
    logger.info(f"Starting long-running task for {duration_minutes} minutes")
    
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    # Simulate work with periodic progress updates
    for i in range(duration_minutes):
        await asyncio.sleep(60)  # Sleep for 1 minute
        elapsed = time.time() - start_time
        logger.info(f"Long-running task progress: {i+1}/{duration_minutes} minutes completed")
    
    total_time = time.time() - start_time
    result = {
        "requested_duration_minutes": duration_minutes,
        "actual_duration_seconds": total_time,
        "status": "success"
    }
    
    logger.info(f"Completed long-running task: {result}")
    return result


def multi_gpu_matmul_task_inner(result_queue: multiprocessing.Queue, gpu_id: int, matrix_size: int, num_iterations: int = 10) -> dict:
    """Inner function for matmul computation on specific GPU"""
    try:
        import torch
        
        # 设置当前GPU
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        
        logger.info(f"GPU {gpu_id}: Starting matmul computation on device {device}")
        
        # 获取GPU信息
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        device_name = torch.cuda.get_device_name(gpu_id)
        props = torch.cuda.get_device_properties(gpu_id)
        
        logger.info(f"GPU {gpu_id}: Device name: {device_name}")
        logger.info(f"GPU {gpu_id}: Compute capability: {props.major}.{props.minor}")
        logger.info(f"GPU {gpu_id}: CUDA_VISIBLE_DEVICES: {visible_devices}")
        
        # 准备计算
        start_time = time.time()
        total_computation_time = 0.0
        
        results_stats = []
        
        for iteration in range(num_iterations):
            # 创建随机矩阵
            a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
            b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
            
            # 执行矩阵乘法
            iter_start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # 确保计算完成
            iter_end = time.time()
            
            iter_time = iter_end - iter_start
            total_computation_time += iter_time
            
            # 记录结果统计
            stats = {
                "iteration": iteration,
                "computation_time": iter_time,
                "result_mean": float(c.mean().cpu()),
                "result_std": float(c.std().cpu())
            }
            results_stats.append(stats)
            
            logger.info(f"GPU {gpu_id}: Iteration {iteration+1}/{num_iterations} completed in {iter_time:.4f}s")
            
            # 清理内存
            del a, b, c
            torch.cuda.empty_cache()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算性能指标
        avg_computation_time = total_computation_time / num_iterations
        throughput = num_iterations / total_time  # iterations per second
        flops_per_iter = 2 * matrix_size**3  # approximate FLOPs for matmul
        total_flops = flops_per_iter * num_iterations
        tflops = total_flops / (total_computation_time * 1e12)  # TFLOPs
        
        result = {
            "gpu_id": gpu_id,
            "device_name": device_name,
            "cuda_visible_devices": visible_devices,
            "compute_capability": f"{props.major}.{props.minor}",
            "matrix_size": matrix_size,
            "num_iterations": num_iterations,
            "total_time": total_time,
            "total_computation_time": total_computation_time,
            "avg_computation_time": avg_computation_time,
            "throughput_iter_per_sec": throughput,
            "estimated_tflops": tflops,
            "memory_allocated_mb": torch.cuda.memory_allocated(gpu_id) / 1024**2,
            "iterations_stats": results_stats,
            "status": "success"
        }
        
        logger.info(f"GPU {gpu_id}: Completed matmul task - Avg time: {avg_computation_time:.4f}s, TFLOPs: {tflops:.2f}")
        result_queue.put(result)
        return result
        
    except Exception as e:
        error_msg = f"GPU {gpu_id} matmul task failed: {e}"
        logger.error(error_msg)
        error_result = {"gpu_id": gpu_id, "status": "failed", "error": str(e)}
        result_queue.put(error_result)
        return error_result


async def concurrent_gpu_benchmark(matrix_sizes: list = None, num_iterations: int = 10000) -> dict:
    """Run concurrent benchmarks on all GPUs with different matrix sizes"""
    if matrix_sizes is None:
        matrix_sizes = [512, 1024, 2048]
    
    try:
        from eval_.common.mprunner import mp_run
        import torch
        
        logger.info(f"Starting concurrent GPU benchmark with sizes: {matrix_sizes}")
        
        if not torch.cuda.is_available():
            return {"status": "failed", "error": "CUDA not available"}
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPUs available")
        
        # 为每个GPU分配不同大小的矩阵进行测试 - 使用异步并发执行
        all_results = []
        
        # 创建所有任务的列表
        async def run_gpu_task(gpu_id: int, size: int) -> dict:
            """异步运行单个GPU任务"""
            result = await asyncio.to_thread(
                mp_run,
                worker_func=multi_gpu_matmul_task_inner,
                args=(gpu_id, size, num_iterations)
            )
            return {
                "gpu_id": gpu_id,
                "matrix_size": size,
                "result": result.result if result.subproc_success else {"status": "failed", "error": result.error}
            }
        
        for size in matrix_sizes:
            logger.info(f"Testing matrix size: {size}x{size}")
            
            # 为当前矩阵大小创建所有GPU的并发任务
            tasks = [run_gpu_task(gpu_id, size) for gpu_id in range(gpu_count)]
            
            # 并发执行所有GPU任务
            size_results = await asyncio.gather(*tasks)
            
            all_results.append({
                "matrix_size": size,
                "gpu_results": size_results
            })
        
        # 生成性能对比报告
        performance_summary = {
            "total_gpus": gpu_count,
            "matrix_sizes_tested": matrix_sizes,
            "num_iterations_per_test": num_iterations,
            "detailed_results": all_results,
            "status": "success"
        }
        
        logger.info(f"Completed concurrent GPU benchmark")
        return performance_summary
        
    except Exception as e:
        logger.error(f"Concurrent GPU benchmark failed: {e}")
        return {"status": "failed", "error": str(e)} 