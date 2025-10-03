#!/usr/bin/env python3
"""Batch submission example: submit multiple tasks and track their progress."""
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
    
    print("=== Batch Task Submission ===\n")
    
    # Define tasks to submit
    tasks = [
        {
            "name": "Functional Test 1",
            "script": "/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            "type": "functional",
            "args": []
        },
        {
            "name": "Memory Test",
            "script": "/workspace/server/nvgpu/test_scripts/memory_stress_test.py",
            "type": "functional",
            "args": ["--size", "5000", "--iterations", "5"]
        },
        {
            "name": "Performance Benchmark",
            "script": "/workspace/server/nvgpu/test_scripts/performance_benchmark.py",
            "type": "performance",
            "args": ["--matmul-size", "2048", "--matmul-iters", "50"]
        },
        {
            "name": "Long Task",
            "script": "/workspace/server/nvgpu/test_scripts/long_running_task.py",
            "type": "functional",
            "args": ["--duration", "30", "--report-interval", "5"]
        }
    ]
    
    # Submit all tasks
    task_ids = []
    for task in tasks:
        print(f"Submitting: {task['name']}")
        task_id = client.submit_task(
            script_path=task["script"],
            task_type=task["type"],
            args=task["args"]
        )
        task_ids.append((task["name"], task_id))
        print(f"  Task ID: {task_id}")
    
    print(f"\nSubmitted {len(task_ids)} tasks\n")
    
    # Monitor progress
    completed = set()
    while len(completed) < len(task_ids):
        print(f"Progress: {len(completed)}/{len(task_ids)} completed")
        
        for name, task_id in task_ids:
            if task_id in completed:
                continue
            
            result = client.get_task(task_id)
            status = result.status
            
            if status == "running":
                print(f"  {name}: Running...")
            elif status == "queued":
                print(f"  {name}: Queued")
            elif status == "pending":
                print(f"  {name}: Pending")
            elif status == "completed":
                print(f"  {name}: ✓ Completed (exit={result.exit_code})")
                completed.add(task_id)
            elif status == "failed":
                print(f"  {name}: ✗ Failed ({result.error_message})")
                completed.add(task_id)
        
        if len(completed) < len(task_ids):
            time.sleep(5)
            print()
    
    # Final summary
    print("\n=== Summary ===")
    success_count = 0
    for name, task_id in task_ids:
        result = client.get_task(task_id)
        if result.status == "completed" and result.exit_code == 0:
            print(f"✓ {name}: SUCCESS")
            success_count += 1
        else:
            print(f"✗ {name}: FAILED")
    
    print(f"\n{success_count}/{len(task_ids)} tasks succeeded")
    return 0 if success_count == len(task_ids) else 1


if __name__ == "__main__":
    sys.exit(main())

