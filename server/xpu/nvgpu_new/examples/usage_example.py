#!/usr/bin/env python3
"""
GPU Management System Usage Examples

This module provides comprehensive examples of how to use the GPU Management System API.
It demonstrates task submission, monitoring, and various system features.
"""

import asyncio
import time
import requests
from typing import Dict, List, Any, Optional

# Configure API endpoint
API_BASE = "http://localhost:8080"


class GPUManagerClient:
    """Client for interacting with the GPU Management API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def submit_task(self, 
                   task_type: str,
                   name: str,
                   description: str,
                   module_path: str,
                   function_name: str,
                   args: List = None,
                   kwargs: Dict = None,
                   max_wait_time_minutes: int = 30,
                   gpu_id: Optional[int] = None) -> str:
        """Submit a task to the GPU management system"""
        
        payload = {
            "task_type": task_type,
            "name": name,
            "description": description,
            "module_path": module_path,
            "function_name": function_name,
            "args": args or [],
            "kwargs": kwargs or {},
            "max_wait_time_minutes": max_wait_time_minutes
        }
        
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        
        response = requests.post(f"{self.base_url}/tasks/submit", json=payload)
        response.raise_for_status()
        
        return response.json()["task_id"]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status"""
        response = requests.get(f"{self.base_url}/gpus")
        response.raise_for_status()
        return response.json()
    
    def get_available_gpus(self) -> Dict[str, Any]:
        """Get available GPU IDs"""
        response = requests.get(f"{self.base_url}/gpus/available")
        response.raise_for_status()
        return response.json()
    
    def get_task_types(self) -> Dict[str, Any]:
        """Get available task types"""
        response = requests.get(f"{self.base_url}/task-types")
        response.raise_for_status()
        return response.json()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            response = requests.delete(f"{self.base_url}/tasks/{task_id}")
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError:
            return False


async def wait_for_task_completion(client: GPUManagerClient, task_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Wait for a task to complete"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = client.get_task_status(task_id)
        
        if status["status"] in ["completed", "failed", "cancelled"]:
            return status
        
        print(f"Task {task_id} status: {status['status']}, waiting...")
        await asyncio.sleep(2)
    
    raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")


async def example_shared_tasks():
    """Example: Submit multiple shared tasks for concurrent execution"""
    print("\n=== Example: Shared Tasks (Concurrent Execution) ===")
    
    client = GPUManagerClient()
    
    # Submit multiple shared tasks
    task_ids = []
    for i in range(3):
        task_id = client.submit_task(
            task_type="shared",
            name=f"Shared Task {i+1}",
            description=f"Simple shared task for concurrent execution test {i+1}",
            module_path="server.xpu.nvgpu_new.examples.simple_tasks",
            function_name="simple_shared_task",
            kwargs={"duration": 3.0, "use_gpu": True}
        )
        task_ids.append(task_id)
        print(f"Submitted shared task {i+1}: {task_id}")
    
    # Wait for all tasks to complete
    results = []
    for task_id in task_ids:
        result = await wait_for_task_completion(client, task_id)
        results.append(result)
        print(f"Task {task_id} completed: {result['status']}")
        if result.get('result'):
            print(f"  Duration: {result['result'].get('duration', 'N/A'):.2f}s")
            print(f"  GPU: {result['result'].get('gpu_device', 'N/A')}")
    
    return results


async def example_exclusive_task():
    """Example: Submit an exclusive task requiring full GPU access"""
    print("\n=== Example: Exclusive Task (Full GPU Access) ===")
    
    client = GPUManagerClient()
    
    # Submit an exclusive task
    task_id = client.submit_task(
        task_type="exclusive",
        name="Exclusive Task",
        description="Intensive exclusive task requiring full GPU access",
        module_path="server.xpu.nvgpu_new.examples.simple_tasks",
        function_name="simple_exclusive_task",
        kwargs={"workload_size": 3000, "iterations": 50}
    )
    
    print(f"Submitted exclusive task: {task_id}")
    
    # Wait for completion
    result = await wait_for_task_completion(client, task_id)
    print(f"Exclusive task completed: {result['status']}")
    
    if result.get('result'):
        print(f"  Duration: {result['result'].get('duration', 'N/A'):.2f}s")
        print(f"  GPU: {result['result'].get('gpu_device', 'N/A')}")
        print(f"  Iterations: {result['result'].get('iterations', 'N/A')}")
    
    return result


async def example_gpu_assignment():
    """Example: Submit task with specific GPU assignment"""
    print("\n=== Example: Specific GPU Assignment ===")
    
    client = GPUManagerClient()
    
    # Get available GPUs
    available_gpus = client.get_available_gpus()
    gpu_ids = available_gpus["available_gpu_ids"]
    
    if not gpu_ids:
        print("No GPUs available for GPU assignment test")
        return
    
    # Try to use the first available GPU
    target_gpu = gpu_ids[0]
    print(f"Requesting specific GPU: {target_gpu}")
    
    task_id = client.submit_task(
        task_type="shared",
        name="GPU Assignment Test",
        description="Testing specific GPU assignment",
        module_path="server.xpu.nvgpu_new.examples.simple_tasks",
        function_name="simple_shared_task",
        kwargs={"duration": 2.0},
        gpu_id=target_gpu
    )
    
    print(f"Submitted task with GPU {target_gpu}: {task_id}")
    
    # Check if it was assigned to the target GPU
    status = client.get_task_status(task_id)
    assigned_gpu = status.get('gpu_id')
    
    if assigned_gpu == target_gpu:
        print(f"✓ Task successfully assigned to GPU {target_gpu}")
    else:
        print(f"⚠ Task assigned to GPU {assigned_gpu} instead of GPU {target_gpu}")
    
    await wait_for_task_completion(client, task_id)


async def example_mixed_task_types():
    """Example: Submit both exclusive and shared tasks to see scheduling behavior"""
    print("\n=== Example: Mixed Task Types (Exclusive + Shared) ===")
    
    client = GPUManagerClient()
    
    # Submit shared tasks first
    shared_task_ids = []
    for i in range(2):
        task_id = client.submit_task(
            task_type="shared",
            name=f"Shared Task {i+1}",
            description="Shared task that can run with others",
            module_path="server.xpu.nvgpu_new.examples.simple_tasks",
            function_name="simple_shared_task",
            kwargs={"duration": 5.0}
        )
        shared_task_ids.append(task_id)
        print(f"Submitted shared task {i+1}: {task_id}")
    
    # Wait a moment for shared tasks to start
    await asyncio.sleep(2)
    
    # Submit an exclusive task
    exclusive_task_id = client.submit_task(
        task_type="exclusive",
        name="Exclusive Task",
        description="Exclusive task that needs full GPU",
        module_path="server.xpu.nvgpu_new.examples.simple_tasks",
        function_name="simple_exclusive_task",
        kwargs={"workload_size": 1000, "iterations": 20}
    )
    print(f"Submitted exclusive task: {exclusive_task_id}")
    
    # Monitor all tasks
    all_task_ids = shared_task_ids + [exclusive_task_id]
    for task_id in all_task_ids:
        result = await wait_for_task_completion(client, task_id)
        print(f"Task {task_id} finished: {result['status']}")


async def example_error_handling():
    """Example: Error handling and recovery"""
    print("\n=== Example: Error Handling ===")
    
    client = GPUManagerClient()
    
    # Submit a task that will fail
    task_id = client.submit_task(
        task_type="shared",
        name="Failing Task",
        description="Task designed to fail for error handling test",
        module_path="server.xpu.nvgpu_new.examples.simple_tasks",
        function_name="failing_task",
        kwargs={"should_fail": True, "error_message": "Intentional test failure"}
    )
    
    print(f"Submitted failing task: {task_id}")
    
    # Wait for it to fail
    result = await wait_for_task_completion(client, task_id)
    
    if result["status"] == "failed":
        print(f"✓ Task failed as expected: {result.get('error', 'No error message')}")
    else:
        print(f"⚠ Task did not fail as expected: {result['status']}")


async def monitor_system_status():
    """Example: Monitor system status"""
    print("\n=== Example: System Status Monitoring ===")
    
    client = GPUManagerClient()
    
    # Get system status
    system_status = client.get_system_status()
    
    print(f"System Status:")
    print(f"  Dispatcher Running: {system_status['dispatcher_running']}")
    print(f"  Total GPUs: {system_status['total_gpus']}")
    print(f"  Available GPUs: {system_status['available_gpu_ids']}")
    print(f"  Configuration: Max parallel tasks = {system_status['configuration']['max_parallel_task_num']}")
    
    # System-wide statistics
    print(f"\nSystem Statistics:")
    print(f"  Global Task Count: {system_status.get('global_task_count', 0)}")
    print(f"  Available GPUs: {system_status.get('available_gpus', 0)}")
    print(f"  Cooldown GPUs: {system_status.get('cooldown_gpus', 0)}")
    print(f"  Unhealthy GPUs: {system_status.get('unhealthy_gpus', 0)}")
    
    # Individual GPU status
    print(f"\nGPU Status:")
    for gpu_id, gpu_info in system_status.get('gpus', {}).items():
        if 'error' in gpu_info:
            print(f"  GPU-{gpu_id}: Error - {gpu_info['error']}")
        else:
            print(f"  GPU-{gpu_id} ({gpu_info.get('name', 'Unknown')}):")
            print(f"    Memory: {gpu_info.get('memory_used_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB")
            print(f"    Utilization: {gpu_info.get('utilization_percent', 0)}%")
            print(f"    Temperature: {gpu_info.get('temperature_c', 0)}°C")
            print(f"    Available: {'Yes' if gpu_info.get('is_available', False) else 'No'}")
            print(f"    In Cooldown: {'Yes' if gpu_info.get('is_in_cooldown', False) else 'No'}")
            
            queue_info = system_status.get('queue_statistics', {}).get(gpu_id, {})
            print(f"    Queue: {queue_info.get('exclusive_queue_size', 0)} exclusive, {queue_info.get('shared_queue_size', 0)} shared")
            print(f"    Running: {'Yes' if queue_info.get('running_exclusive') else 'No'} exclusive, {queue_info.get('running_shared_count', 0)} shared")


async def main():
    """Main example execution"""
    print("GPU Management System - Usage Examples")
    print("======================================")
    
    client = GPUManagerClient()
    
    try:
        # Check if server is running
        system_status = client.get_system_status()
        print(f"Connected to GPU Management System (v4.0)")
        print(f"Available GPUs: {system_status['total_gpus']}")
        
        # Get available task types
        task_types = client.get_task_types()
        print(f"Available task types: {list(task_types['available_types'])}")
        
        # Run examples
        await monitor_system_status()
        await example_shared_tasks()
        await example_exclusive_task()
        await example_gpu_assignment()
        await example_mixed_task_types()
        await example_error_handling()
        
        print("\n=== All Examples Completed ===")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to GPU Management API server")
        print("Please ensure the server is running on http://localhost:8080")
        print("Start the server with: python start_server.py")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 