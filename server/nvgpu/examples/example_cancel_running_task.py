#!/usr/bin/env python3
"""
Example: Cancel running tasks

Demonstrates:
- Cancelling pending/queued tasks
- Force-cancelling running tasks
- Handling different cancellation scenarios
"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from client import NVGPUClient
import time

def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("Server is not running!")
        return 1
    
    print("=== Task Cancellation Example ===\n")
    
    # Test 1: Cancel a pending/queued task (normal cancel)
    print("Test 1: Cancelling a queued task (normal)...")
    task_id = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
        task_type="functional",
        args=["--duration", "300"]  # 5 minutes
    )
    print(f"  Task submitted: {task_id}")
    
    # Wait a moment to ensure it's queued/running
    time.sleep(0.5)
    
    # Try normal cancel
    if client.cancel_task(task_id, force=False):
        print("  ✓ Task cancelled (normal)\n")
    else:
        print("  ✗ Normal cancel failed (task may be running)\n")
        # Clean up with force cancel
        client.cancel_task(task_id, force=True)
    
    # Test 2: Force cancel a running task
    print("Test 2: Force-cancelling a running task...")
    task_id = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
        task_type="functional",
        args=["--duration", "300"]  # 5 minutes
    )
    print(f"  Task submitted: {task_id}")
    
    # Wait for task to start running
    time.sleep(2)
    
    result = client.get_task(task_id)
    print(f"  Task status: {result.status}")
    
    if result.status == "running":
        print("  Attempting force cancel...")
        if client.cancel_task(task_id, force=True):
            print("  ✓ Running task force-cancelled successfully")
            
            # Verify cancellation
            time.sleep(1)
            result = client.get_task(task_id)
            print(f"  Final status: {result.status}")
            print(f"  Error message: {result.error_message}\n")
        else:
            print("  ✗ Force cancel failed\n")
    else:
        print(f"  Task is not running (status: {result.status}), skipping\n")
        if result.status == "queued":
            client.cancel_task(task_id, force=False)
    
    # Test 3: Try to force cancel a task that doesn't exist
    print("Test 3: Cancelling non-existent task...")
    try:
        client.cancel_task("non-existent-task-id", force=True)
        print("  ✗ Should have failed\n")
    except Exception as e:
        print(f"  ✓ Correctly failed: Request failed\n")
    
    # Test 4: Cancel multiple running tasks
    print("Test 4: Batch cancellation of running tasks...")
    task_ids = []
    for i in range(3):
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "300"]
        )
        task_ids.append(task_id)
        print(f"  Submitted task {i+1}: {task_id[:8]}...")
    
    # Wait for tasks to start
    time.sleep(3)
    
    # Cancel all tasks
    print("  Cancelling all tasks...")
    cancelled_count = 0
    for task_id in task_ids:
        if client.cancel_task(task_id, force=True):
            cancelled_count += 1
    
    print(f"  ✓ Cancelled {cancelled_count}/{len(task_ids)} tasks\n")
    
    # Verify all are cancelled
    time.sleep(1)
    for i, task_id in enumerate(task_ids):
        result = client.get_task(task_id)
        print(f"  Task {i+1}: {result.status}")
    
    print("\n=== Cancellation Tests Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())

