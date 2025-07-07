#!/usr/bin/env python3
"""
Kill Task and CUDA Error Recovery Example

This example demonstrates:
1. Real task killing (not just cancellation)
2. CUDA error automatic task killing and re-queuing
3. Task recovery after cooldown and health check
4. Manual kill task API
5. Difference between cancel and kill operations

New Features:
- Kill running tasks (not just cancel pending ones)
- CUDA error kills all running tasks immediately
- Killed tasks are automatically re-queued after recovery
- Manual kill task API endpoint
"""

import asyncio
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def long_running_task(duration: int = 30):
    """Simulate a long-running task that can be killed"""
    print(f"🚀 Starting long-running task (duration: {duration}s)")
    
    for i in range(duration):
        print(f"⏳ Task progress: {i+1}/{duration} seconds")
        await asyncio.sleep(1)
        
        # Check if we should raise CUDA error halfway through
        if i == duration // 2:
            # This simulates a CUDA error occurring during execution
            pass  # Will be controlled externally
    
    print("✅ Long-running task completed successfully")
    return f"Task completed after {duration} seconds"

async def cuda_error_task():
    """Task that will cause CUDA error after a few seconds"""
    print("🔥 Starting task that will cause CUDA error...")
    await asyncio.sleep(3)  # Run for 3 seconds before error
    print("💥 Triggering CUDA error!")
    raise RuntimeError("CUDA out of memory: Tried to allocate 12.00 GiB")

async def fast_task():
    """A fast task for testing"""
    print("⚡ Running fast task...")
    await asyncio.sleep(2)
    return "Fast task completed"

async def submit_task_via_api(task_type: str, task_name: str, function_name: str, 
                             duration: Optional[int] = None, gpu_id: Optional[int] = None,
                             base_url: str = "http://localhost:8080"):
    """Submit task via API"""
    try:
        import requests
        
        payload = {
            "task_type": task_type,
            "name": task_name,
            "description": f"Test task: {task_name}",
            "module_path": "server.xpu.nvgpu_new.examples.kill_task_example",
            "function_name": function_name,
            "args": [duration] if duration else [],
            "kwargs": {}
        }
        
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        
        response = requests.post(f"{base_url}/tasks/submit", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task submitted: {result['task_id']}")
            return result['task_id']
        else:
            print(f"❌ Task submission failed: {response.json()['detail']}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to submit task: {e}")
        return None

async def get_task_status(task_id: str, base_url: str = "http://localhost:8080"):
    """Get task status"""
    try:
        import requests
        response = requests.get(f"{base_url}/tasks/{task_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"❌ Failed to get task status: {e}")
        return None

async def kill_task_via_api(task_id: str, base_url: str = "http://localhost:8080"):
    """Kill task via API"""
    try:
        import requests
        response = requests.delete(f"{base_url}/tasks/{task_id}/kill")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🔪 Task killed: {task_id}")
            return True
        else:
            print(f"❌ Kill failed: {response.json()['detail']}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to kill task: {e}")
        return False

async def cancel_task_via_api(task_id: str, base_url: str = "http://localhost:8080"):
    """Cancel task via API"""
    try:
        import requests
        response = requests.delete(f"{base_url}/tasks/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"❌ Task cancelled: {task_id}")
            return True
        else:
            print(f"❌ Cancel failed: {response.json()['detail']}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to cancel task: {e}")
        return False

async def monitor_gpu_status(base_url: str = "http://localhost:8080"):
    """Monitor GPU status"""
    try:
        import requests
        response = requests.get(f"{base_url}/status")
        status = response.json()
        
        print("\n📊 GPU Status:")
        for gpu_id, gpu_status in status['gpus'].items():
            print(f"  GPU-{gpu_id}:")
            print(f"    Available: {gpu_status['is_available']}")
            print(f"    In Cooldown: {gpu_status['is_in_cooldown']}")
            
            # Get queue info from queue_statistics
            queue_info = status.get('queue_statistics', {}).get(gpu_id, {})
            has_running = queue_info.get('running_exclusive', False) or queue_info.get('running_shared_count', 0) > 0
            print(f"    Running Tasks: {has_running}")
            print(f"    Queue Size: Exclusive={queue_info.get('exclusive_queue_size', 0)}, Shared={queue_info.get('shared_queue_size', 0)}")
            
            if gpu_status['is_in_cooldown']:
                print(f"    Cooldown Reason: {gpu_status['cooldown_reason']}")
                print(f"    Cooldown Expires: {gpu_status['cooldown_expires_at']}")
            
    except Exception as e:
        print(f"❌ Failed to get GPU status: {e}")

async def wait_for_task_completion(task_id: str, max_wait: int = 60, base_url: str = "http://localhost:8080"):
    """Wait for task completion with status updates"""
    print(f"⏳ Monitoring task {task_id[:8]}...")
    
    for i in range(max_wait):
        status = await get_task_status(task_id, base_url)
        if status:
            if status['status'] in ['completed', 'failed', 'cancelled', 'killed']:
                print(f"📋 Task {task_id[:8]}: {status['status']}")
                if status['error']:
                    print(f"   Error: {status['error'][:100]}...")
                return status
            elif i % 5 == 0:  # Print update every 5 seconds
                print(f"   Status: {status['status']} (execution: {status['execution_time_seconds']:.1f}s)")
                
        await asyncio.sleep(1)
    
    print(f"⏰ Task {task_id[:8]} monitoring timeout")
    return None

async def main():
    """Main demonstration"""
    print("=== Kill Task and CUDA Error Recovery Example ===")
    print()
    
    base_url = "http://localhost:8080"
    
    # Check if server is running
    try:
        import requests
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server not running. Please start the server first:")
            print("   python start_server.py --error-cooldown 30")
            return
    except Exception:
        print("❌ Server not running. Please start the server first:")
        print("   python start_server.py --error-cooldown 30")
        return
    
    print("✅ Server is running. Starting demonstrations...\n")
    
    # 1. Basic Task Kill Demo
    print("=== 1. Basic Task Kill Demonstration ===")
    
    # Submit a long-running shared task
    task_id = await submit_task_via_api(
        task_type="shared",
        task_name="Long Running Task",
        function_name="long_running_task",
        duration=30
    )
    
    if task_id:
        # Wait for task to start running
        await asyncio.sleep(3)
        
        # Check status
        status = await get_task_status(task_id)
        if status and status['status'] == 'running':
            print(f"📋 Task {task_id[:8]} is running. Killing it...")
            
            # Kill the task
            killed = await kill_task_via_api(task_id)
            
            if killed:
                # Wait for kill to take effect
                await asyncio.sleep(2)
                final_status = await get_task_status(task_id)
                if final_status:
                    print(f"📋 Final status: {final_status['status']}")
                    print(f"   Execution time: {final_status['execution_time_seconds']:.1f}s")
    
    print("\n" + "="*50)
    
    # 2. Cancel vs Kill Demo
    print("=== 2. Cancel vs Kill Comparison ===")
    
    # Submit two tasks, one to cancel, one to kill
    cancel_task_id = await submit_task_via_api(
        task_type="shared",
        task_name="Task to Cancel",
        function_name="long_running_task",
        duration=20
    )
    
    kill_task_id = await submit_task_via_api(
        task_type="shared",
        task_name="Task to Kill",
        function_name="long_running_task",
        duration=20
    )
    
    if cancel_task_id and kill_task_id:
        # Wait for tasks to start
        await asyncio.sleep(3)
        
        # Cancel first task (only works if pending)
        print(f"🔄 Trying to cancel task {cancel_task_id[:8]}...")
        await cancel_task_via_api(cancel_task_id)
        
        # Kill second task (works for running tasks)
        print(f"🔪 Killing task {kill_task_id[:8]}...")
        await kill_task_via_api(kill_task_id)
        
        # Monitor both
        await wait_for_task_completion(cancel_task_id, max_wait=10)
        await wait_for_task_completion(kill_task_id, max_wait=10)
    
    print("\n" + "="*50)
    
    # 3. CUDA Error Recovery Demo
    print("=== 3. CUDA Error Recovery Demonstration ===")
    
    # Submit a task that causes CUDA error
    cuda_task_id = await submit_task_via_api(
        task_type="exclusive",  # Use exclusive for more dramatic effect
        task_name="CUDA Error Task",
        function_name="cuda_error_task"
    )
    
    if cuda_task_id:
        print(f"📋 CUDA error task submitted: {cuda_task_id[:8]}")
        print("⏳ Waiting for CUDA error to occur...")
        
        # Monitor for error and recovery
        result = await wait_for_task_completion(cuda_task_id, max_wait=30)
        
        if result and result['status'] == 'failed':
            print("💥 CUDA error occurred as expected!")
            print("⏳ Monitoring GPU recovery...")
            
            # Monitor GPU status during recovery
            for i in range(10):
                await monitor_gpu_status()
                await asyncio.sleep(5)
                print(f"   Recovery check {i+1}/10")
                
                # Check if GPU is recovered
                import requests
                status_response = requests.get(f"{base_url}/status")
                if status_response.status_code == 200:
                    system_status = status_response.json()
                    available_count = system_status.get('available_gpus', 0)
                    cooldown_count = system_status.get('cooldown_gpus', 0)
                    
                    if available_count > 0 and cooldown_count == 0:
                        print("✅ GPU recovered! All GPUs are available again.")
                        break
    
    print("\n" + "="*50)
    
    # 4. Multiple Task Kill Demo
    print("=== 4. Multiple Task Kill Demonstration ===")
    
    # Submit multiple shared tasks
    task_ids = []
    for i in range(3):
        task_id = await submit_task_via_api(
            task_type="shared",
            task_name=f"Parallel Task {i+1}",
            function_name="long_running_task",
            duration=25
        )
        if task_id:
            task_ids.append(task_id)
    
    if task_ids:
        print(f"📋 Submitted {len(task_ids)} parallel tasks")
        
        # Wait for tasks to start
        await asyncio.sleep(3)
        
        # Kill all tasks
        print("🔪 Killing all tasks...")
        for task_id in task_ids:
            await kill_task_via_api(task_id)
        
        # Monitor completion
        for task_id in task_ids:
            await wait_for_task_completion(task_id, max_wait=10)
    
    print("\n=== Kill Task Demonstration Complete ===")
    print("Key takeaways:")
    print("- Kill works on running tasks, cancel works on pending tasks")
    print("- CUDA errors trigger automatic task killing and GPU cooldown")
    print("- System automatically recovers after cooldown and health check")
    print("- Multiple tasks can be killed simultaneously")

if __name__ == "__main__":
    asyncio.run(main()) 