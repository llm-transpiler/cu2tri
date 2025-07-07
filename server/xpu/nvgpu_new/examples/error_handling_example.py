#!/usr/bin/env python3
"""
CUDA Error Handling and Health Check Example

This example demonstrates:
1. CUDA error detection and cooldown mechanism
2. Error-driven health check system
3. Automatic recovery after cooldown
4. Manual health check via API
5. Task safety during health checks

The system no longer runs periodic health checks, but instead:
- Detects CUDA errors during task execution
- Enters cooldown mode (default: 30 seconds)
- Performs health check only after cooldown expires
- Extends cooldown if health check fails
- Ensures no tasks are running during health checks
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def simulate_cuda_error():
    """Simulate a CUDA error"""
    print("🔥 Simulating CUDA error...")
    raise RuntimeError("CUDA out of memory: Tried to allocate 8.00 GiB")

async def simulate_healthy_task():
    """Simulate a healthy task"""
    print("✅ Executing healthy task...")
    await asyncio.sleep(2)
    return "Task completed successfully"

async def simulate_driver_error():
    """Simulate a CUDA driver error"""
    print("🔥 Simulating CUDA driver error...")
    raise RuntimeError("CUDA driver error: device disconnected")

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
            print(f"    Health Check Passed: {gpu_status['last_health_check_passed']}")
            print(f"    Error Count: {gpu_status['cuda_error_count']}")
            print(f"    Cooldown Reason: {gpu_status['cooldown_reason'] or 'None'}")
            if gpu_status['is_in_cooldown']:
                print(f"    Cooldown Expires: {gpu_status['cooldown_expires_at']}")
            
    except Exception as e:
        print(f"❌ Failed to get GPU status: {e}")

async def trigger_manual_health_check(gpu_id: int, base_url: str = "http://localhost:8080"):
    """Trigger manual health check"""
    try:
        import requests
        response = requests.post(f"{base_url}/gpus/{gpu_id}/health-check")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Manual health check for GPU-{gpu_id}: {'PASSED' if result['health_check_passed'] else 'FAILED'}")
        elif response.status_code == 409:
            print(f"⚠️  Cannot run health check on GPU-{gpu_id}: {response.json()['detail']}")
        else:
            print(f"❌ Health check failed: {response.json()['detail']}")
            
    except Exception as e:
        print(f"❌ Failed to trigger health check: {e}")

async def submit_task_via_api(task_type: str, task_name: str, gpu_id: Optional[int] = None,
                             base_url: str = "http://localhost:8080"):
    """Submit task via API"""
    try:
        import requests
        
        # Determine which function to call based on task name
        if "error" in task_name.lower():
            if "driver" in task_name.lower():
                function_name = "simulate_driver_error"
            else:
                function_name = "simulate_cuda_error"
        else:
            function_name = "simulate_healthy_task"
            
        payload = {
            "task_type": task_type,
            "name": task_name,
            "description": f"Test task: {task_name}",
            "module_path": "server.xpu.nvgpu_new.examples.error_handling_example",
            "function_name": function_name,
            "args": [],
            "kwargs": {},
            "gpu_id": gpu_id
        }
        
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

async def wait_for_task_completion(task_id: str, base_url: str = "http://localhost:8080"):
    """Wait for task completion"""
    try:
        import requests
        
        for _ in range(30):  # Wait up to 30 seconds
            response = requests.get(f"{base_url}/tasks/{task_id}")
            
            if response.status_code == 200:
                status = response.json()
                if status['status'] in ['completed', 'failed']:
                    print(f"📋 Task {task_id}: {status['status']}")
                    if status['error']:
                        print(f"   Error: {status['error'][:100]}...")
                    return status
                    
            await asyncio.sleep(1)
            
        print(f"⏰ Task {task_id} did not complete within 30 seconds")
        return None
        
    except Exception as e:
        print(f"❌ Failed to check task status: {e}")
        return None

async def main():
    """Main demonstration"""
    print("=== CUDA Error Handling and Health Check Example ===")
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
    
    print("✅ Server is running")
    print()
    
    # 1. Show initial system status
    print("1. Initial System Status:")
    await monitor_gpu_status(base_url)
    print()
    
    # 2. Submit a healthy task first
    print("2. Submit healthy task:")
    task_id = await submit_task_via_api("exclusive", "Healthy Task")
    if task_id:
        await wait_for_task_completion(task_id, base_url)
    print()
    
    # 3. Submit a task that will cause CUDA error
    print("3. Submit task that causes CUDA error:")
    task_id = await submit_task_via_api("exclusive", "CUDA Error Task")
    if task_id:
        await wait_for_task_completion(task_id, base_url)
    print()
    
    # 4. Show GPU status after error (should be in cooldown)
    print("4. GPU Status after CUDA error:")
    await monitor_gpu_status(base_url)
    print()
    
    # 5. Try to submit another task (should fail due to cooldown)
    print("5. Try to submit task during cooldown:")
    task_id = await submit_task_via_api("exclusive", "Task during cooldown")
    print()
    
    # 6. Try manual health check during cooldown
    print("6. Try manual health check during cooldown:")
    await trigger_manual_health_check(0, base_url)
    print()
    
    # 7. Wait for cooldown to expire
    print("7. Waiting for cooldown to expire (30 seconds)...")
    print("   (The system will automatically perform health check after cooldown)")
    
    for i in range(30):
        print(f"   Cooldown: {30-i} seconds remaining...", end='\r')
        await asyncio.sleep(1)
    print("\n   Cooldown period completed!")
    print()
    
    # 8. Wait a bit more for automatic health check
    print("8. Waiting for automatic health check...")
    await asyncio.sleep(5)
    
    # 9. Check GPU status after cooldown
    print("9. GPU Status after cooldown (should show health check result):")
    await monitor_gpu_status(base_url)
    print()
    
    # 10. Try manual health check when GPU is idle
    print("10. Manual health check when GPU is idle:")
    await trigger_manual_health_check(0, base_url)
    print()
    
    # 11. Submit another healthy task to verify recovery
    print("11. Submit healthy task to verify recovery:")
    task_id = await submit_task_via_api("exclusive", "Recovery Test Task")
    if task_id:
        await wait_for_task_completion(task_id, base_url)
    print()
    
    # 12. Final system status
    print("12. Final System Status:")
    await monitor_gpu_status(base_url)
    print()
    
    print("=== Error Handling Demonstration Complete ===")
    print()
    print("Key Features Demonstrated:")
    print("✓ CUDA error detection and automatic cooldown")
    print("✓ Error-driven health check (not periodic)")
    print("✓ Task safety during health checks")
    print("✓ Manual health check API")
    print("✓ Automatic recovery after successful health check")
    print("✓ Cooldown extension when health check fails")
    
    print()
    print("New Cooldown Mechanism:")
    print("- Default cooldown: 30 seconds (vs. 5 minutes before)")
    print("- Health check only runs after cooldown expires")
    print("- GPU must be idle for health check to run")
    print("- Failed health check extends cooldown period")
    print("- No periodic health checks (reduces overhead)")

if __name__ == "__main__":
    asyncio.run(main()) 