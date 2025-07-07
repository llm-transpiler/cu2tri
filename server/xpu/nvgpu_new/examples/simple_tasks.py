"""
Simple Example Tasks for GPU Management System

These are basic examples to demonstrate shared and exclusive task types.
"""

import asyncio
import time
import random
import torch
from typing import Any, Dict, List, Optional


async def simple_gpu_task(duration: float = 2.0, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Simple GPU task for testing - simplified version
    
    Args:
        duration: How long to run (seconds)
        use_gpu: Whether to use GPU operations
    
    Returns:
        Task result dictionary
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting simple GPU task (duration: {duration}s)")
    
    if use_gpu and torch.cuda.is_available():
        # Simple GPU computation
        device = torch.cuda.current_device()
        print(f"[Task-{task_id}] Using GPU-{device}")
        
        # Create tensors and do basic operations
        x = torch.randn(2000, 2000, device=f'cuda:{device}')
        
        # Simulate work
        steps = int(duration * 10)
        for i in range(steps):
            x = torch.matmul(x, x.T)
            x = torch.relu(x)
            x = x / (x.norm() + 1e-8)  # Normalize to prevent overflow
            await asyncio.sleep(0.1)
        
        result = {
            'task_id': task_id,
            'gpu_device': device,
            'duration': time.time() - start_time,
            'steps_completed': steps
        }
    else:
        # CPU-only computation
        print(f"[Task-{task_id}] Using CPU")
        await asyncio.sleep(duration)
        
        result = {
            'task_id': task_id,
            'gpu_device': 'cpu',
            'duration': time.time() - start_time
        }
    
    print(f"[Task-{task_id}] Completed in {result['duration']:.2f}s")
    return result


async def simple_shared_task(duration: float = 2.0, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Simple shared task for testing concurrent execution
    
    Args:
        duration: How long to run (seconds)
        use_gpu: Whether to use GPU operations
    
    Returns:
        Task result dictionary
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting shared task (duration: {duration}s)")
    
    if use_gpu and torch.cuda.is_available():
        # Simple GPU computation
        device = torch.cuda.current_device()
        print(f"[Task-{task_id}] Using GPU-{device}")
        
        # Create some tensors and do basic operations
        x = torch.randn(1000, 1000, device=f'cuda:{device}')
        y = torch.randn(1000, 1000, device=f'cuda:{device}')
        
        # Simulate work by doing matrix operations
        for i in range(int(duration * 10)):  # Adjust workload based on duration
            z = torch.matmul(x, y)
            x = torch.relu(z)
            await asyncio.sleep(0.1)  # Allow other tasks to run
        
        result = {
            'task_id': task_id,
            'task_type': 'shared',
            'gpu_device': device,
            'tensor_shape': x.shape,
            'final_sum': float(x.sum()),
            'duration': time.time() - start_time
        }
    else:
        # CPU-only computation
        print(f"[Task-{task_id}] Using CPU")
        
        # Simulate work
        result_value = 0
        for i in range(int(duration * 100)):
            result_value += random.random()
            await asyncio.sleep(0.01)
        
        result = {
            'task_id': task_id,
            'task_type': 'shared',
            'gpu_device': 'cpu',
            'computation_result': result_value,
            'duration': time.time() - start_time
        }
    
    print(f"[Task-{task_id}] Completed shared task in {result['duration']:.2f}s")
    return result


async def simple_exclusive_task(workload_size: int = 5000, iterations: int = 100) -> Dict[str, Any]:
    """
    Simple exclusive task requiring full GPU access
    
    Args:
        workload_size: Size of tensors to create
        iterations: Number of iterations to run
    
    Returns:
        Task result dictionary
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting exclusive task (full GPU access)")
    
    if not torch.cuda.is_available():
        raise RuntimeError("Exclusive task requires CUDA GPU")
    
    device = torch.cuda.current_device()
    print(f"[Task-{task_id}] Using GPU-{device} exclusively")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Allocate large tensors to simulate exclusive usage
    tensors = []
    for i in range(4):  # Create multiple large tensors
        tensor = torch.randn(workload_size, workload_size, device=f'cuda:{device}')
        tensors.append(tensor)
    
    # Perform intensive computation
    results = []
    for i in range(iterations):
        # Matrix operations between all tensor pairs
        for j in range(len(tensors)):
            for k in range(j+1, len(tensors)):
                result_tensor = torch.matmul(tensors[j], tensors[k])
                results.append(float(result_tensor.sum()))
        
        if i % 10 == 0:
            print(f"[Task-{task_id}] Progress: {i}/{iterations} iterations")
            await asyncio.sleep(0.1)  # Brief pause
    
    # Compute final statistics
    final_result = {
        'task_id': task_id,
        'task_type': 'exclusive',
        'gpu_device': device,
        'workload_size': workload_size,
        'iterations': iterations,
        'tensor_count': len(tensors),
        'results_count': len(results),
        'mean_result': sum(results) / len(results) if results else 0,
        'max_result': max(results) if results else 0,
        'min_result': min(results) if results else 0,
        'duration': time.time() - start_time
    }
    
    # Clean up
    del tensors
    del results
    torch.cuda.empty_cache()
    
    print(f"[Task-{task_id}] Completed exclusive task in {final_result['duration']:.2f}s")
    return final_result


# Backward compatibility aliases
async def simple_functional_task(duration: float = 2.0, use_gpu: bool = True) -> Dict[str, Any]:
    """Backward compatibility alias for simple_shared_task"""
    result = await simple_shared_task(duration, use_gpu)
    result['task_type'] = 'functional'  # Maintain legacy type
    return result


async def simple_performance_task(workload_size: int = 5000, iterations: int = 100) -> Dict[str, Any]:
    """Backward compatibility alias for simple_exclusive_task"""
    result = await simple_exclusive_task(workload_size, iterations)
    result['task_type'] = 'performance'  # Maintain legacy type
    return result


async def cpu_intensive_task(duration: float = 5.0) -> Dict[str, Any]:
    """
    CPU-intensive task that doesn't use GPU
    
    Args:
        duration: How long to run (seconds)
    
    Returns:
        Task result dictionary
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting CPU intensive task (duration: {duration}s)")
    
    # Simulate CPU work with some computation
    total = 0
    steps = int(duration * 1000)  # Many small steps
    
    for i in range(steps):
        # Some CPU work
        total += sum(range(100))
        
        if i % 100 == 0:  # Allow other tasks to run
            await asyncio.sleep(0.001)
    
    result = {
        'task_id': task_id,
        'task_type': 'cpu_intensive',
        'computation_result': total,
        'steps_completed': steps,
        'duration': time.time() - start_time
    }
    
    print(f"[Task-{task_id}] Completed CPU task in {result['duration']:.2f}s")
    return result


async def memory_test_task(memory_mb: int = 1000) -> Dict[str, Any]:
    """
    Task that tests GPU memory allocation
    
    Args:
        memory_mb: Amount of GPU memory to allocate (MB)
    
    Returns:
        Task result dictionary
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting memory test task ({memory_mb}MB)")
    
    if not torch.cuda.is_available():
        raise RuntimeError("Memory test requires CUDA GPU")
    
    device = torch.cuda.current_device()
    
    # Calculate tensor size for desired memory allocation
    # Float32 tensors use 4 bytes per element
    elements_per_mb = (1024 * 1024) // 4
    total_elements = memory_mb * elements_per_mb
    
    # Create square tensor
    side_length = int(total_elements ** 0.5)
    
    try:
        print(f"[Task-{task_id}] Allocating {memory_mb}MB on GPU-{device}")
        large_tensor = torch.randn(side_length, side_length, device=f'cuda:{device}')
        
        # Do some computation to verify allocation worked
        result_tensor = torch.sum(large_tensor)
        result_value = float(result_tensor)
        
        # Hold the memory for a moment
        await asyncio.sleep(2.0)
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        
        result = {
            'task_id': task_id,
            'task_type': 'memory_test',
            'gpu_device': device,
            'memory_allocated_mb': memory_mb,
            'tensor_shape': (side_length, side_length),
            'computation_result': result_value,
            'duration': time.time() - start_time,
            'success': True
        }
        
    except RuntimeError as e:
        print(f"[Task-{task_id}] Memory allocation failed: {e}")
        result = {
            'task_id': task_id,
            'task_type': 'memory_test',
            'gpu_device': device,
            'memory_requested_mb': memory_mb,
            'error': str(e),
            'duration': time.time() - start_time,
            'success': False
        }
    
    print(f"[Task-{task_id}] Memory test completed in {result['duration']:.2f}s")
    return result


def failing_task(should_fail: bool = True, error_message: str = "Simulated task failure") -> Dict[str, Any]:
    """
    Task that intentionally fails for testing error handling
    
    Args:
        should_fail: Whether to actually fail
        error_message: Error message to raise
    
    Returns:
        Task result dictionary (or raises exception)
    """
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting failing task")
    
    if should_fail:
        print(f"[Task-{task_id}] Raising error: {error_message}")
        raise RuntimeError(error_message)
    else:
        return {
            'task_id': task_id,
            'task_type': 'failing_test',
            'message': 'Task completed without failing',
            'duration': 0.1
        }


async def cuda_error_simulation() -> Dict[str, Any]:
    """
    Task that simulates a CUDA error for testing error handling
    
    Returns:
        Task result dictionary (or raises CUDA exception)
    """
    start_time = time.time()
    task_id = random.randint(1000, 9999)
    
    print(f"[Task-{task_id}] Starting CUDA error simulation")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA error simulation requires CUDA GPU")
    
    device = torch.cuda.current_device()
    print(f"[Task-{task_id}] Using GPU-{device} to simulate CUDA error")
    
    # Try to allocate way too much memory to trigger CUDA OOM error
    try:
        # Attempt to allocate an impossibly large tensor
        huge_tensor = torch.randn(100000, 100000, device=f'cuda:{device}')
        
        # If somehow this succeeds, clean up and return
        del huge_tensor
        return {
            'task_id': task_id,
            'task_type': 'cuda_error_simulation',
            'gpu_device': device,
            'result': 'Unexpectedly succeeded',
            'duration': time.time() - start_time
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"[Task-{task_id}] Successfully triggered CUDA error: {e}")
            # Re-raise the CUDA error for the system to handle
            raise e
        else:
            # Some other error, re-raise it
            raise e


def get_task_function(task_name: str):
    """Helper function to get task function by name"""
    task_functions = {
        'simple_gpu_task': simple_gpu_task,
        'simple_shared_task': simple_shared_task,
        'simple_exclusive_task': simple_exclusive_task,
        'simple_functional_task': simple_functional_task,  # Legacy alias
        'simple_performance_task': simple_performance_task,  # Legacy alias
        'cpu_intensive_task': cpu_intensive_task,
        'memory_test_task': memory_test_task,
        'failing_task': failing_task,
        'cuda_error_simulation': cuda_error_simulation,
    }
    
    return task_functions.get(task_name) 