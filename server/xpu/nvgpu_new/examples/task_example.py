#!/usr/bin/env python3
"""
GPU Management System - Task Types Example

This example demonstrates how to use exclusive and shared tasks.
"""

import asyncio
import requests
import time
import json
from typing import Dict, Any, List


# API base URL
API_BASE = "http://localhost:8080"


def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.status_code == 200
    except:
        return False


def get_system_status():
    """Get and display system status"""
    response = requests.get(f"{API_BASE}/status")
    status = response.json()
    
    print("\n=== System Status ===")
    print(f"Total GPUs: {status['total_gpus']}")
    print(f"Available GPUs: {status['available_gpu_ids']}")
    print(f"Max Parallel Tasks: {status['configuration']['max_parallel_task_num']}")
    
    # Calculate total statistics from queue statistics
    total_exclusive_queued = 0
    total_shared_queued = 0
    total_running_exclusive = 0
    total_running_shared = 0
    
    for gpu_id, queue_stats in status['queue_statistics'].items():
        total_exclusive_queued += queue_stats.get('exclusive_queue_size', 0)
        total_shared_queued += queue_stats.get('shared_queue_size', 0)
        total_running_exclusive += 1 if queue_stats.get('running_exclusive') else 0
        total_running_shared += queue_stats.get('running_shared_count', 0)
    
    print("\nTask Statistics:")
    print(f"  Exclusive queued: {total_exclusive_queued}")
    print(f"  Shared queued: {total_shared_queued}")
    print(f"  Running exclusive: {total_running_exclusive}")
    print(f"  Running shared: {total_running_shared}")
    
    print("\nGPU Details:")
    for gpu_id, gpu_info in status['gpus'].items():
        queue_stats = status['queue_statistics'].get(gpu_id, {})
        print(f"  GPU {gpu_id} ({gpu_info.get('name', 'Unknown')})")
        print(f"    Memory: {gpu_info.get('memory_used_mb', 0)}/{gpu_info.get('memory_total_mb', 0)} MB")
        print(f"    Exclusive queue: {queue_stats.get('exclusive_queue_size', 0)}")
        print(f"    Shared queue: {queue_stats.get('shared_queue_size', 0)}")
        print(f"    Running: {1 if queue_stats.get('running_exclusive') else 0} exclusive, "
              f"{queue_stats.get('running_shared_count', 0)} shared")


def submit_task(task_type: str, name: str, duration: float, gpu_id: int = None) -> str:
    """Submit a task to the system"""
    task_data = {
        "task_type": task_type,
        "name": name,
        "description": f"{task_type.capitalize()} task running for {duration}s",
        "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
        "function_name": "simple_gpu_task",
        "kwargs": {"duration": duration, "use_gpu": True}
    }
    
    if gpu_id is not None:
        task_data["gpu_id"] = gpu_id
    
    response = requests.post(f"{API_BASE}/tasks/submit", json=task_data)
    if response.status_code == 200:
        return response.json()["task_id"]
    else:
        print(f"Failed to submit task: {response.text}")
        return None


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status"""
    response = requests.get(f"{API_BASE}/tasks/{task_id}")
    if response.status_code == 200:
        return response.json()
    return None


def wait_for_task(task_id: str, timeout: int = 60) -> Dict[str, Any]:
    """Wait for a task to complete"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = get_task_status(task_id)
        if status and status['status'] in ['completed', 'failed', 'cancelled']:
            return status
        time.sleep(1)
    
    return None


def demo_exclusive_tasks():
    """Demonstrate exclusive task execution"""
    print("\n=== Exclusive Tasks Demo ===")
    print("Exclusive tasks require full GPU access - only one can run at a time")
    
    # Submit multiple exclusive tasks
    task_ids = []
    for i in range(3):
        task_id = submit_task("exclusive", f"Exclusive Task {i+1}", duration=3.0)
        if task_id:
            task_ids.append(task_id)
            print(f"Submitted exclusive task {i+1}: {task_id}")
    
    # Monitor execution
    print("\nMonitoring tasks (should run one at a time)...")
    for i in range(12):  # Monitor for ~12 seconds
        exclusive_running = 0
        for task_id in task_ids:
            status = get_task_status(task_id)
            if status and status['status'] == 'running':
                exclusive_running += 1
        print(f"  Time {i}s: {exclusive_running} exclusive task(s) running")
        time.sleep(1)
    
    # Wait for all to complete
    print("\nWaiting for all tasks to complete...")
    for task_id in task_ids:
        status = wait_for_task(task_id)
        if status:
            print(f"  Task {status['name']}: {status['status']} "
                  f"(GPU {status['gpu_id']}, {status['execution_time_seconds']:.1f}s)")


def demo_shared_tasks():
    """Demonstrate shared task execution"""
    print("\n=== Shared Tasks Demo ===")
    print("Shared tasks can run in parallel on the same GPU")
    
    # Submit multiple shared tasks
    task_ids = []
    for i in range(6):
        task_id = submit_task("shared", f"Shared Task {i+1}", duration=4.0)
        if task_id:
            task_ids.append(task_id)
            print(f"Submitted shared task {i+1}: {task_id}")
    
    # Monitor execution
    print("\nMonitoring tasks (multiple should run in parallel)...")
    max_parallel = 0
    for i in range(8):  # Monitor for ~8 seconds
        shared_running = 0
        for task_id in task_ids:
            status = get_task_status(task_id)
            if status and status['status'] == 'running':
                shared_running += 1
        max_parallel = max(max_parallel, shared_running)
        print(f"  Time {i}s: {shared_running} shared task(s) running")
        time.sleep(1)
    
    print(f"\nMaximum parallel shared tasks observed: {max_parallel}")
    
    # Wait for all to complete
    print("\nWaiting for all tasks to complete...")
    for task_id in task_ids:
        status = wait_for_task(task_id)
        if status:
            print(f"  Task {status['name']}: {status['status']} "
                  f"(GPU {status['gpu_id']}, {status['execution_time_seconds']:.1f}s)")


def demo_mixed_tasks():
    """Demonstrate mixed exclusive and shared task scheduling"""
    print("\n=== Mixed Tasks Demo ===")
    print("Showing how exclusive and shared tasks interact")
    
    # Submit shared tasks first
    shared_ids = []
    for i in range(2):
        task_id = submit_task("shared", f"Shared Task {i+1}", duration=5.0)
        if task_id:
            shared_ids.append(task_id)
            print(f"Submitted shared task {i+1}: {task_id}")
    
    time.sleep(2)  # Let shared tasks start
    
    # Submit an exclusive task
    exclusive_id = submit_task("exclusive", "Exclusive Task", duration=3.0)
    print(f"Submitted exclusive task: {exclusive_id}")
    
    # Submit more shared tasks
    for i in range(2, 4):
        task_id = submit_task("shared", f"Shared Task {i+1}", duration=3.0)
        if task_id:
            shared_ids.append(task_id)
            print(f"Submitted shared task {i+1}: {task_id}")
    
    # Monitor execution
    print("\nMonitoring task execution...")
    print("(Exclusive task should wait for running shared tasks to complete)")
    
    all_task_ids = shared_ids + [exclusive_id]
    for i in range(15):  # Monitor for ~15 seconds
        statuses = []
        for task_id in all_task_ids:
            status = get_task_status(task_id)
            if status and status['status'] == 'running':
                statuses.append(f"{status['task_type']}:{status['name']}")
        
        if statuses:
            print(f"  Time {i}s: Running - {', '.join(statuses)}")
        else:
            print(f"  Time {i}s: No tasks running")
        
        time.sleep(1)
    
    # Final status
    print("\nFinal task status:")
    for task_id in all_task_ids:
        status = get_task_status(task_id)
        if status:
            print(f"  {status['name']} ({status['task_type']}): "
                  f"{status['status']} in {status['execution_time_seconds']:.1f}s")


def demo_gpu_assignment():
    """Demonstrate direct GPU assignment"""
    print("\n=== GPU Assignment Demo ===")
    print("Tasks can be assigned to specific GPUs")
    
    # Get available GPUs
    response = requests.get(f"{API_BASE}/gpus/available")
    available_gpus = response.json()['available_gpu_ids']
    
    if len(available_gpus) < 2:
        print("Need at least 2 GPUs for this demo")
        return
    
    print(f"Available GPUs: {available_gpus}")
    
    # Submit tasks to specific GPUs
    task_assignments = []
    
    # Exclusive task on GPU 0
    task_id = submit_task("exclusive", "GPU-0 Exclusive", duration=3.0, gpu_id=available_gpus[0])
    if task_id:
        task_assignments.append((task_id, available_gpus[0], "exclusive"))
        print(f"Submitted exclusive task to GPU {available_gpus[0]}: {task_id}")
    
    # Shared tasks on GPU 1
    for i in range(2):
        task_id = submit_task("shared", f"GPU-1 Shared {i+1}", duration=3.0, gpu_id=available_gpus[1])
        if task_id:
            task_assignments.append((task_id, available_gpus[1], "shared"))
            print(f"Submitted shared task to GPU {available_gpus[1]}: {task_id}")
    
    # Monitor GPU assignments
    print("\nChecking GPU assignments...")
    time.sleep(1)
    
    for task_id, expected_gpu, task_type in task_assignments:
        status = get_task_status(task_id)
        if status:
            actual_gpu = status['gpu_id']
            if actual_gpu == expected_gpu:
                print(f"  ✓ {task_type} task {task_id} correctly assigned to GPU {actual_gpu}")
            else:
                print(f"  ✗ {task_type} task {task_id} on GPU {actual_gpu} (expected {expected_gpu})")
    
    # Wait for completion
    print("\nWaiting for tasks to complete...")
    for task_id, _, _ in task_assignments:
        wait_for_task(task_id)


def main():
    """Main demo function"""
    print("=== GPU Management System - Task Types Demo ===")
    
    # Check server
    if not check_server():
        print("\nERROR: Server is not running!")
        print("Please start the server with: python start_server.py")
        return
    
    print("\nServer is running. Starting demos...")
    
    # Get initial status
    get_system_status()
    
    # Run demos
    try:
        demo_exclusive_tasks()
        demo_shared_tasks()
        demo_mixed_tasks()
        demo_gpu_assignment()
    except Exception as e:
        print(f"\nError during demo: {e}")
    
    # Final status
    print("\n" + "="*50)
    get_system_status()


if __name__ == "__main__":
    main() 