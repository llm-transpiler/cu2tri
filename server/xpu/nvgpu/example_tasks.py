"""
Example tasks for GPU Management API

Shows how to define custom tasks that can be submitted to the GPU management system.
"""

import asyncio
import time
import logging
import os

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
    try:
        import torch
        
        logger.info(f"Starting GPU compute task with n={n}")
        
        # Create random tensors on GPU
        a = torch.randn(n, n, device='cuda')
        b = torch.randn(n, n, device='cuda')
        
        # Perform matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        end_time = time.time()
        
        # Get result statistics
        result = {
            "matrix_size": n,
            "computation_time": end_time - start_time,
            "result_mean": float(c.mean().cpu()),
            "result_std": float(c.std().cpu()),
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "status": "success"
        }
        
        logger.info(f"Completed GPU compute task: {result}")
        return result
        
    except Exception as e:
        logger.error(f"GPU compute task failed: {e}")
        return {"status": "failed", "error": str(e)}


def memory_intensive_task(memory_gb: float = 1.0) -> dict:
    """A task that allocates significant GPU memory"""
    try:
        import torch
        
        logger.info(f"Starting memory intensive task with {memory_gb}GB")
        
        # Calculate tensor size for desired memory usage
        bytes_per_element = 4  # float32
        elements_needed = int(memory_gb * 1024**3 / bytes_per_element)
        tensor_size = int(elements_needed**0.5)  # Square matrix
        
        # Allocate memory
        tensor = torch.randn(tensor_size, tensor_size, device='cuda')
        
        # Simple computation
        result_tensor = tensor @ tensor.T
        
        # Get statistics
        result = {
            "requested_memory_gb": memory_gb,
            "actual_tensor_size": [tensor_size, tensor_size],
            "actual_memory_gb": torch.cuda.memory_allocated() / 1024**3,
            "computation_result_mean": float(result_tensor.mean().cpu()),
            "status": "success"
        }
        
        logger.info(f"Completed memory intensive task: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Memory intensive task failed: {e}")
        return {"status": "failed", "error": str(e)}


def check_gpu_assignment() -> dict:
    """Check which GPU this task is actually running on using subprocess"""
    try:
        from eval_.common.mprunner import mp_run
        result = mp_run(check_gpu_assignment_inner)
        logger.info(f"GPU assignment check result: {result}")
        return result.result
    except Exception as e:
        logger.error(f"GPU assignment check failed: {e}")
        return {"status": "failed", "error": str(e)}
import multiprocessing
def check_gpu_assignment_inner(result_queue: multiprocessing.Queue) -> dict:
    """Check which GPU this task is actually running on"""
    try:
        import torch
        # Get CUDA_VISIBLE_DEVICES setting
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}") # 不知道为什么永远好像指向第一个DEVICE, 而且改了代码直接发起请求好像还是用的原来的代码
        '''
        2025-06-22 21:01:16,335 - server.xpu.nvgpu.example_tasks - INFO - Starting GPU compute task with n=100
2025-06-22 21:01:16,914 - server.xpu.nvgpu.example_tasks - INFO - Completed GPU compute task: {'matrix_size': 100, 'computation_time': 0.07730770111083984, 'result_mean': -0.010694164782762527, 'result_std': 9.991312026977539, 'gpu_memory_allocated': 33675776, 'status': 'success'}
[INFO] - Task 95e3f275-c650-4399-8eab-a2caed173d99 completed on GPU 2 after 2.20s
[INFO] - Executing task 1ed6ee6e-98fe-45c1-bdfd-9fac7f1dfbf3 on GPU 5
2025-06-22 21:01:16,915 - server.xpu.nvgpu.example_tasks - INFO - CUDA_VISIBLE_DEVICES: 5
2025-06-22 21:01:16,989 - server.xpu.nvgpu.example_tasks - INFO - Compute capability: 9.0
2025-06-22 21:01:16,989 - server.xpu.nvgpu.example_tasks - INFO - device_name: NVIDIA H100 PCIe
2025-06-22 21:01:16,989 - server.xpu.nvgpu.example_tasks - INFO - GPU assignment check: {'cuda_visible_devices': '5', 'current_device_id': 0, 'device_name': 'NVIDIA H100 PCIe', 'gpu_type': 'Unknown', 'total_visible_devices': 1, 'status': 'success'}
[INFO] - Task 1ed6ee6e-98fe-45c1-bdfd-9fac7f1dfbf3 completed on GPU 5 after 2.24s
[INFO] - Executing task 662df6a2-6117-4e37-aa60-1bd98676b202 on GPU 0
2025-06-22 21:01:16,989 - server.xpu.nvgpu.example_tasks - INFO - CUDA_VISIBLE_DEVICES: 0
2025-06-22 21:01:17,050 - server.xpu.nvgpu.example_tasks - INFO - Compute capability: 9.0
2025-06-22 21:01:17,050 - server.xpu.nvgpu.example_tasks - INFO - device_name: NVIDIA H100 PCIe
2025-06-22 21:01:17,050 - server.xpu.nvgpu.example_tasks - INFO - GPU assignment check: {'cuda_visible_devices': '0', 'current_device_id': 0, 'device_name': 'NVIDIA H100 PCIe', 'gpu_type': 'Unknown', 'total_visible_devices': 1, 'status': 'success'}
[INFO] - Task 662df6a2-6117-4e37-aa60-1bd98676b202 completed on GPU 0 after 2.29s
[INFO] - Executing task 24adfef6-6e0f-46e8-bc83-9d28aa23bac8 on GPU 0
2025-06-22 21:01:17,050 - server.xpu.nvgpu.example_tasks - INFO - CUDA_VISIBLE_DEVICES: 0
2025-06-22 21:01:17,110 - server.xpu.nvgpu.example_tasks - INFO - Compute capability: 9.0
2025-06-22 21:01:17,110 - server.xpu.nvgpu.example_tasks - INFO - device_name: NVIDIA H100 PCIe
2025-06-22 21:01:17,110 - server.xpu.nvgpu.example_tasks - INFO - GPU assignment check: {'cuda_visible_devices': '0', 'current_device_id': 0, 'device_name': 'NVIDIA H100 PCIe', 'gpu_type': 'Unknown', 'total_visible_devices': 1, 'status': 'success'}
[INFO] - Task 24adfef6-6e0f-46e8-bc83-9d28aa23bac8 completed on GPU 0 after 2.34s
[INFO] - Executing task 299ea6a5-87fa-4a28-9e1f-1b17d5260157 on GPU 1
2025-06-22 21:01:17,110 - server.xpu.nvgpu.example_tasks - INFO - CUDA_VISIBLE_DEVICES: 1
2025-06-22 21:01:17,171 - server.xpu.nvgpu.example_tasks - INFO - Compute capability: 9.0
2025-06-22 21:01:17,171 - server.xpu.nvgpu.example_tasks - INFO - device_name: NVIDIA H100 PCIe
2025-06-22 21:01:17,171 - server.xpu.nvgpu.example_tasks - INFO - GPU assignment check: {'cuda_visible_devices': '1', 'current_device_id': 0, 'device_name': 'NVIDIA H100 PCIe', 'gpu_type': 'Unknown', 'total_visible_devices': 1, 'status': 'success'}
[INFO] - Task 299ea6a5-87fa-4a28-9e1f-1b17d5260157 completed on GPU 1 after 2.38s
[INFO] - Executing task 6c00f009-b65e-4036-94ec-8b64d810e760 on GPU 3
2025-06-22 21:01:17,171 - server.xpu.nvgpu.example_tasks - INFO - CUDA_VISIBLE_DEVICES: 3
2025-06-22 21:01:17,241 - server.xpu.nvgpu.example_tasks - INFO - Compute capability: 9.0
2025-06-22 21:01:17,242 - server.xpu.nvgpu.example_tasks - INFO - device_name: NVIDIA H100 PCIe
2025-06-22 21:01:17,242 - server.xpu.nvgpu.example_tasks - INFO - GPU assignment check: {'cuda_visible_devices': '3', 'current_device_id': 0, 'device_name': 'NVIDIA H100 PCIe', 'gpu_type': 'Unknown', 'total_visible_devices': 1, 'status': 'success'}
        '''
        
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
            try:
                test_tensor1 = torch.tensor([1.0, 2.0], device=current_device)
                test_tensor2 = torch.tensor([3.0, 4.0], device=current_device)
                for i in range(10000):
                    result = test_tensor1 + test_tensor2
                    # print(f"  GPU functionality test: PASSED {i}")
                del test_tensor1, test_tensor2, result
                torch.cuda.empty_cache()
            except Exception as test_e:
                print(f"  GPU functionality test: FAILED - {str(test_e)}")
            
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