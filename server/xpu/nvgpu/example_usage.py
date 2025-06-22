"""
Example usage of GPU Manager for kernel testing tasks

This example demonstrates how to use the GPU management system for
running kernel comparison tasks.
"""

import asyncio
import logging
from pathlib import Path

from .gpu_manager import GPUManager
from .task_queue import TaskType
from eval_.kernelbench_c.gpu_task_executor import (
    create_correctness_task,
    create_performance_task,
    create_comprehensive_task
)
from utils.set_env import PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example"""
    logger.info("=== Basic GPU Manager Usage Example ===")
    
    # Create and start GPU manager
    async with GPUManager(refresh_interval_seconds=5) as gpu_manager:
        
        # Check system status
        status = await gpu_manager.get_system_status()
        logger.info(f"System has {len(status['gpus'])} GPUs available")
        
        # Submit a functional task (correctness comparison)
        kernel_dir = PROJECT_ROOT / "cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_"
        
        task_id = await gpu_manager.submit_task(
            task_type=TaskType.FUNCTIONAL,
            name="Matrix Multiplication Correctness Test",
            description="Compare Triton vs PyTorch correctness for matrix multiplication",
            execute_func=create_correctness_task,
            args=(kernel_dir, "triton_vs_torch", "example_test"),
            estimated_memory_mb=2000,
            max_wait_time_minutes=10
        )
        
        logger.info(f"Submitted functional task: {task_id}")
        
        # Monitor task progress
        while True:
            task_status = await gpu_manager.get_task_status(task_id)
            if not task_status:
                break
                
            logger.info(f"Task {task_id} status: {task_status['status']}")
            
            if task_status['status'] in ['completed', 'failed', 'cancelled']:
                if task_status['status'] == 'completed':
                    logger.info(f"Task completed successfully on GPU {task_status['gpu_id']}")
                else:
                    logger.error(f"Task failed: {task_status.get('error', 'Unknown error')}")
                break
            
            await asyncio.sleep(2)


async def example_performance_task():
    """Performance task example"""
    logger.info("=== Performance Task Example ===")
    
    async with GPUManager() as gpu_manager:
        
        # Submit a performance task (requires exclusive GPU)
        kernel_dir = "cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_"
        
        task_id = await gpu_manager.submit_task(
            task_type=TaskType.PERFORMANCE,
            name="Matrix Multiplication Performance Test",
            description="Compare Triton vs CUDA performance for matrix multiplication",
            execute_func=create_performance_task,
            args=(kernel_dir, "triton_vs_cuda", "perf_test"),
            estimated_memory_mb=8000,
            max_wait_time_minutes=15
        )
        
        logger.info(f"Submitted performance task: {task_id}")
        
        # Monitor task
        while True:
            task_status = await gpu_manager.get_task_status(task_id)
            if not task_status:
                break
                
            logger.info(f"Performance task status: {task_status['status']}")
            
            if task_status['status'] in ['completed', 'failed', 'cancelled']:
                if task_status['status'] == 'completed':
                    logger.info(f"Performance task completed on GPU {task_status['gpu_id']}")
                    logger.info(f"Execution time: {task_status['execution_time_seconds']:.2f}s")
                break
            
            await asyncio.sleep(3)


async def example_comprehensive_task():
    """Comprehensive comparison task example"""
    logger.info("=== Comprehensive Comparison Example ===")
    
    async with GPUManager() as gpu_manager:
        
        # Submit comprehensive task
        kernel_dir = "cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_"
        
        task_id = await gpu_manager.submit_task(
            task_type=TaskType.PERFORMANCE,  # Comprehensive tasks are performance tasks
            name="Comprehensive Matrix Multiplication Test",
            description="Full comparison of all kernel implementations",
            execute_func=create_comprehensive_task,
            args=(kernel_dir, "comprehensive_test"),
            estimated_memory_mb=12000,
            max_wait_time_minutes=30
        )
        
        logger.info(f"Submitted comprehensive task: {task_id}")
        
        # Monitor with detailed logging
        while True:
            task_status = await gpu_manager.get_task_status(task_id)
            if not task_status:
                break
            
            queue_status = await gpu_manager.get_queue_status()
            logger.info(f"Comprehensive task: {task_status['status']}, "
                       f"Queue: {queue_status['performance_queue_length']} performance, "
                       f"{queue_status['functional_queue_length']} functional")
            
            if task_status['status'] in ['completed', 'failed', 'cancelled']:
                if task_status['status'] == 'completed':
                    logger.info(f"Comprehensive task completed successfully!")
                    logger.info(f"Total execution time: {task_status['execution_time_seconds']:.2f}s")
                break
            
            await asyncio.sleep(5)


async def example_multiple_tasks():
    """Example with multiple concurrent tasks"""
    logger.info("=== Multiple Tasks Example ===")
    
    async with GPUManager() as gpu_manager:
        
        # Get available GPUs
        gpu_status = await gpu_manager.get_gpu_status()
        logger.info(f"Available GPUs: {list(gpu_status.keys())}")
        
        # Submit multiple tasks
        tasks = []
        kernel_dirs = [
            "cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_",
            # Add more kernel directories as available
        ]
        
        for i, kernel_dir in enumerate(kernel_dirs):
            # Submit functional tasks
            for comparison in ["triton_vs_torch", "triton_vs_cuda", "cuda_vs_torch"]:
                task_id = await gpu_manager.submit_task(
                    task_type=TaskType.FUNCTIONAL,
                    name=f"Correctness Test {i+1} - {comparison}",
                    description=f"Correctness comparison for {comparison}",
                    execute_func=create_correctness_task,
                    args=(kernel_dir, comparison, f"multi_test_{i}_{comparison}"),
                    estimated_memory_mb=1500
                )
                tasks.append(task_id)
                logger.info(f"Submitted functional task: {task_id}")
            
            # Submit one performance task
            perf_task_id = await gpu_manager.submit_task(
                task_type=TaskType.PERFORMANCE,
                name=f"Performance Test {i+1}",
                description=f"Performance comparison for kernel {i+1}",
                execute_func=create_performance_task,
                args=(kernel_dir, "triton_vs_cuda", f"multi_perf_{i}"),
                estimated_memory_mb=6000
            )
            tasks.append(perf_task_id)
            logger.info(f"Submitted performance task: {perf_task_id}")
        
        # Monitor all tasks
        completed_tasks = set()
        while len(completed_tasks) < len(tasks):
            queue_status = await gpu_manager.get_queue_status()
            logger.info(f"Queue status - Running: {queue_status['running_tasks_count']}, "
                       f"Functional queue: {queue_status['functional_queue_length']}, "
                       f"Performance queue: {queue_status['performance_queue_length']}")
            
            for task_id in tasks:
                if task_id in completed_tasks:
                    continue
                    
                task_status = await gpu_manager.get_task_status(task_id)
                if task_status and task_status['status'] in ['completed', 'failed', 'cancelled']:
                    completed_tasks.add(task_id)
                    logger.info(f"Task {task_id} finished with status: {task_status['status']}")
            
            await asyncio.sleep(5)
        
        logger.info("All tasks completed!")


async def example_gpu_selection():
    """Example of GPU selection and visibility control"""
    logger.info("=== GPU Selection Example ===")
    
    async with GPUManager() as gpu_manager:
        
        # Get all GPUs
        all_gpus = await gpu_manager.get_gpu_status()
        logger.info(f"All available GPUs: {list(all_gpus.keys())}")
        
        # Filter H100 GPUs
        h100_gpus = [
            int(gpu_id) for gpu_id, info in all_gpus.items() 
            if info['gpu_type'] == 'H100'
        ]
        
        if h100_gpus:
            logger.info(f"Found H100 GPUs: {h100_gpus}")
            
            # Set visible GPUs to H100 only
            gpu_manager.set_visible_gpus(h100_gpus)
            
            # Submit task that will run on H100
            task_id = await gpu_manager.submit_task(
                task_type=TaskType.PERFORMANCE,
                name="H100 Performance Test",
                description="Performance test specifically on H100 GPUs",
                execute_func=create_performance_task,
                args=("cu2tri/outputs/kernelbench_c/1_Square_matrix_multiplication_", 
                      "triton_vs_cuda", "h100_test"),
                estimated_memory_mb=8000
            )
            
            logger.info(f"Submitted H100-specific task: {task_id}")
            
            # Monitor task
            while True:
                task_status = await gpu_manager.get_task_status(task_id)
                if not task_status:
                    break
                    
                if task_status['status'] in ['completed', 'failed', 'cancelled']:
                    if task_status['status'] == 'completed':
                        logger.info(f"H100 task completed on GPU {task_status['gpu_id']}")
                    break
                
                await asyncio.sleep(2)
        else:
            logger.info("No H100 GPUs found")


async def main():
    """Run all examples"""
    examples = [
        example_basic_usage,
        example_performance_task,
        example_comprehensive_task,
        example_multiple_tasks,
        example_gpu_selection
    ]
    
    for example_func in examples:
        try:
            await example_func()
            logger.info(f"✅ {example_func.__name__} completed successfully")
        except Exception as e:
            logger.error(f"❌ {example_func.__name__} failed: {e}")
        
        logger.info("-" * 50)
        await asyncio.sleep(2)  # Brief pause between examples


if __name__ == "__main__":
    asyncio.run(main()) 