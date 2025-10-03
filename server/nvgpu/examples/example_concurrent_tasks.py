#!/usr/bin/env python3
"""Example demonstrating concurrent task limits on shared GPUs."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("ERROR: Server not responding")
        return 1
    
    print("=== GPU Concurrent Task Limit Example ===\n")
    
    # Get GPU info
    gpus = client.list_gpus()
    if not gpus:
        print("No GPUs available")
        return 1
    
    gpu = gpus[0]
    gpu_id = gpu['gpu_id']
    
    print(f"GPU {gpu_id} Configuration:")
    print(f"  Mode: {gpu['mode']}")
    print(f"  Max Concurrent Tasks: {gpu['max_concurrent_tasks']}")
    print(f"  Memory Threshold: {gpu['memory_threshold']*100:.0f}%")
    print(f"  Current Running Tasks: {gpu['running_task_count']}")
    print()
    
    # Set a low max_concurrent_tasks for testing
    print("Setting max_concurrent_tasks to 2 for testing...")
    if client.set_gpu_max_concurrent_tasks(gpu_id, 2):
        print("✓ Max concurrent tasks set to 2\n")
    else:
        print("✗ Failed to set max concurrent tasks\n")
        return 1
    
    # Submit 5 long-running tasks
    print("Submitting 5 long-running tasks...")
    task_ids = []
    for i in range(5):
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "30", "--report-interval", "5"],
            gpu_id=gpu_id
        )
        task_ids.append(task_id)
        print(f"  Task {i+1}: {task_id}")
        time.sleep(0.5)  # Small delay between submissions
    
    print()
    
    # Monitor task states
    print("Monitoring task states...")
    for _ in range(10):  # Monitor for 10 iterations
        time.sleep(2)
        
        gpu = client.get_gpu(gpu_id)
        
        status_counts = {"pending": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0}
        for task_id in task_ids:
            task = client.get_task(task_id)
            status_counts[task.status] += 1
        
        print(f"GPU {gpu_id}: Running={gpu['running_task_count']}/{gpu['max_concurrent_tasks']} | "
              f"Pending={status_counts['pending']}, Queued={status_counts['queued']}, "
              f"Running={status_counts['running']}, Completed={status_counts['completed']}")
        
        if status_counts['running'] == 0 and status_counts['pending'] == 0 and status_counts['queued'] == 0:
            break
    
    print()
    
    # Wait for all tasks to complete
    print("Waiting for all tasks to complete...")
    for task_id in task_ids:
        try:
            result = client.wait_for_task(task_id, timeout=60)
            status = "✓" if result.status == "completed" else "✗"
            print(f"  {status} Task {task_id[:8]}: {result.status}")
        except TimeoutError:
            print(f"  ✗ Task {task_id[:8]}: timed out")
    
    # Restore default max_concurrent_tasks
    print("\nRestoring max_concurrent_tasks to 3...")
    if client.set_gpu_max_concurrent_tasks(gpu_id, 3):
        print("✓ Restored to default (3)")
    
    print("\n=== Demonstration Complete ===")
    print("\nKey Points:")
    print("- Only 2 tasks ran concurrently (max_concurrent_tasks=2)")
    print("- Other tasks waited in queue")
    print("- This prevents GPU overload from tasks that gradually use memory")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

