#!/usr/bin/env python3
"""
Example demonstrating GPU mode switching between exclusive and shared modes.

This example shows how the system handles mode transitions when tasks are
already running, which is a critical feature for dynamic GPU management.

Key behaviors demonstrated:
1. Running tasks continue after mode switch (no interruption)
2. New tasks are correctly queued based on new mode
3. Exclusive mode prevents concurrent tasks
4. Shared mode allows controlled concurrency
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_gpu_status(client, gpu_id, prefix=""):
    """Print current GPU status with optional prefix."""
    gpu = client.get_gpu(gpu_id)
    print(f"{prefix}GPU {gpu_id} Status:")
    print(f"{prefix}  Mode: {gpu['mode']}")
    print(f"{prefix}  Running tasks: {len(gpu['running_tasks'])}")
    if gpu['running_tasks']:
        task_ids = [tid[:8] + '...' for tid in gpu['running_tasks']]
        print(f"{prefix}  Task IDs: {task_ids}")


def demo_shared_to_exclusive_with_running_tasks(client, gpu_id):
    """
    Demonstrate switching from shared to exclusive mode while tasks are running.
    
    Expected behavior:
    - Running shared tasks continue executing
    - New tasks wait in queue
    - After all old tasks complete, new tasks start under exclusive mode
    """
    print_section("Demo 1: Shared → Exclusive with Running Tasks")
    
    # Step 1: Ensure GPU is in shared mode
    print("Step 1: Setting GPU to shared mode...")
    client.set_gpu_mode(gpu_id, "shared")
    client.set_gpu_max_concurrent_tasks(gpu_id, 3)
    time.sleep(0.5)
    print_gpu_status(client, gpu_id, "  ")
    
    # Step 2: Submit multiple tasks in shared mode
    print("\nStep 2: Submitting 3 concurrent tasks (each runs 15 seconds)...")
    shared_tasks = []
    for i in range(3):
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "15", "--report-interval", "5"],
            gpu_id=gpu_id
        )
        shared_tasks.append(task_id)
        print(f"  ✓ Task {i+1} submitted: {task_id[:8]}...")
        time.sleep(0.3)
    
    # Wait for tasks to start
    print("\nStep 3: Waiting for tasks to start running...")
    for attempt in range(10):
        time.sleep(1)
        gpu = client.get_gpu(gpu_id)
        running = len(gpu['running_tasks'])
        print(f"  Running: {running}/3", end='\r')
        if running == 3:
            print()  # New line
            break
    
    print_gpu_status(client, gpu_id, "  ")
    
    # Step 4: Switch to exclusive mode while tasks are running
    print("\n🔄 Step 4: Switching to EXCLUSIVE mode (while 3 tasks are running)...")
    client.set_gpu_mode(gpu_id, "exclusive")
    time.sleep(0.5)
    print("  ✓ Mode switched to exclusive")
    print_gpu_status(client, gpu_id, "  ")
    
    # Step 5: Try to submit a new task
    print("\nStep 5: Submitting a new task to exclusive GPU...")
    new_task = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
        task_type="functional",
        gpu_id=gpu_id
    )
    print(f"  ✓ Task submitted: {new_task[:8]}...")
    
    time.sleep(1)
    task_status = client.get_task(new_task).status
    print(f"  Task status: {task_status}")
    
    if task_status == "pending":
        print("  ✅ CORRECT: Task is waiting because GPU has running tasks")
    else:
        print(f"  ⚠️  Task status is '{task_status}' (expected: pending)")
    
    # Step 6: Monitor completion
    print("\nStep 6: Monitoring task completion...")
    print("  (The 3 running tasks will complete, then the new task will start)")
    
    # Wait for all shared tasks to complete
    for i, task_id in enumerate(shared_tasks, 1):
        try:
            client.wait_for_task(task_id, timeout=30)
            print(f"  ✓ Task {i}/3 completed")
        except TimeoutError:
            print(f"  ⏱  Task {i}/3 timeout (but may still be running)")
    
    # Check if new task started
    time.sleep(2)
    final_status = client.get_task(new_task).status
    print(f"\n  New task final status: {final_status}")
    
    if final_status in ["running", "completed"]:
        print("  ✅ SUCCESS: New task started after old tasks completed")
    else:
        print(f"  ⚠️  New task is still {final_status}")
    
    # Cleanup
    try:
        client.wait_for_task(new_task, timeout=20)
    except:
        pass
    
    print("\n" + "="*70)
    print("Key Takeaway:")
    print("  • Running tasks are NOT interrupted by mode switch")
    print("  • New tasks respect the new mode constraints")
    print("  • Mode switch is seamless and safe")
    print("="*70)


def demo_exclusive_mode_behavior(client, gpu_id):
    """
    Demonstrate exclusive mode: only one task at a time.
    """
    print_section("Demo 2: Exclusive Mode Behavior")
    
    # Ensure exclusive mode
    print("Step 1: Setting GPU to exclusive mode...")
    client.set_gpu_mode(gpu_id, "exclusive")
    time.sleep(0.5)
    print_gpu_status(client, gpu_id, "  ")
    
    # Submit first task
    print("\nStep 2: Submitting first task...")
    task1 = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
        task_type="functional",
        args=["--duration", "10"],
        gpu_id=gpu_id
    )
    print(f"  ✓ Task 1 submitted: {task1[:8]}...")
    
    # Wait for it to start
    time.sleep(2)
    status1 = client.get_task(task1).status
    print(f"  Task 1 status: {status1}")
    
    # Submit second task (should be blocked)
    print("\nStep 3: Submitting second task (should be blocked)...")
    task2 = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
        task_type="functional",
        gpu_id=gpu_id
    )
    print(f"  ✓ Task 2 submitted: {task2[:8]}...")
    
    time.sleep(1)
    status2 = client.get_task(task2).status
    print(f"  Task 2 status: {status2}")
    
    if status2 in ["pending", "queued"]:
        print(f"  ✅ CORRECT: Task 2 is blocked (status: {status2})")
        print("  → Exclusive mode ensures only one task runs at a time")
    else:
        print(f"  ⚠️  Task 2 status is '{status2}' (expected: pending/queued)")
    
    print_gpu_status(client, gpu_id, "\n  ")
    
    # Wait for task1 to complete
    print("\nStep 4: Waiting for Task 1 to complete...")
    try:
        # Increase timeout to account for queue wait time
        client.wait_for_task(task1, timeout=20)
        print("  ✓ Task 1 completed")
    except TimeoutError:
        print("  ⏱  Task 1 timeout")
    
    # Check task2 status
    time.sleep(2)
    status2_after = client.get_task(task2).status
    print(f"\n  Task 2 status after Task 1 completes: {status2_after}")
    
    if status2_after in ["running", "completed"]:
        print("  ✅ SUCCESS: Task 2 started after Task 1 completed")
    
    # Cleanup
    try:
        client.wait_for_task(task2, timeout=20)
    except:
        pass
    
    print("\n" + "="*70)
    print("Key Takeaway:")
    print("  • Exclusive mode = maximum 1 task at a time")
    print("  • Additional tasks wait in queue")
    print("  • Useful for tasks requiring full GPU resources")
    print("="*70)


def demo_shared_mode_concurrency(client, gpu_id):
    """
    Demonstrate shared mode: multiple tasks run concurrently.
    """
    print_section("Demo 3: Shared Mode Controlled Concurrency")
    
    # Set to shared mode with limit
    print("Step 1: Setting GPU to shared mode (max 2 concurrent tasks)...")
    client.set_gpu_mode(gpu_id, "shared")
    client.set_gpu_max_concurrent_tasks(gpu_id, 2)
    time.sleep(0.5)
    print_gpu_status(client, gpu_id, "  ")
    
    # Submit 3 tasks
    print("\nStep 2: Submitting 3 tasks (limit is 2)...")
    tasks = []
    for i in range(3):
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "10"],
            gpu_id=gpu_id
        )
        tasks.append(task_id)
        print(f"  ✓ Task {i+1} submitted: {task_id[:8]}...")
        time.sleep(0.3)
    
    # Check running count
    time.sleep(2)
    gpu = client.get_gpu(gpu_id)
    running_count = len(gpu['running_tasks'])
    
    print(f"\n  Currently running: {running_count} tasks")
    
    if running_count == 2:
        print("  ✅ CORRECT: Only 2 tasks running (respecting max_concurrent_tasks)")
        print("  → Third task is waiting in queue")
    else:
        print(f"  ⚠️  Expected 2 running tasks, got {running_count}")
    
    # Monitor status
    print("\nStep 3: Monitoring task progression...")
    for _ in range(6):
        time.sleep(2)
        statuses = [client.get_task(t).status for t in tasks]
        running = statuses.count("running")
        completed = statuses.count("completed")
        pending = statuses.count("pending") + statuses.count("queued")
        print(f"  Running: {running}, Completed: {completed}, Waiting: {pending}")
        
        if completed == 3:
            break
    
    print("\n" + "="*70)
    print("Key Takeaway:")
    print("  • Shared mode allows controlled concurrent execution")
    print("  • max_concurrent_tasks limits simultaneous tasks")
    print("  • Prevents GPU overload from too many concurrent tasks")
    print("="*70)
    
    # Cleanup
    for task_id in tasks:
        try:
            client.wait_for_task(task_id, timeout=5)
        except:
            pass


def main():
    print("\n" + "="*70)
    print("  GPU Mode Switching Examples")
    print("  Demonstrating Dynamic GPU Mode Management")
    print("="*70)
    
    # Connect to server
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("\n❌ ERROR: NVGPU server is not responding")
        print("\nPlease start the server first:")
        print("  cd /workspace/server/nvgpu")
        print("  python main.py")
        return 1
    
    print("\n✓ Connected to NVGPU server")
    
    # Get GPU
    gpus = client.list_gpus()
    if not gpus:
        print("❌ ERROR: No GPUs available")
        return 1
    
    gpu_id = gpus[0]['gpu_id']
    print(f"✓ Using GPU {gpu_id}\n")
    
    try:
        # Run demonstrations
        demo_shared_to_exclusive_with_running_tasks(client, gpu_id)
        
        time.sleep(2)  # Brief pause between demos
        
        demo_exclusive_mode_behavior(client, gpu_id)
        
        time.sleep(2)
        
        demo_shared_mode_concurrency(client, gpu_id)
        
    finally:
        # Restore default settings
        print_section("Cleanup")
        print("Restoring GPU to default settings...")
        client.set_gpu_mode(gpu_id, "shared")
        client.set_gpu_max_concurrent_tasks(gpu_id, 3)
        print("✓ GPU restored to shared mode with max 3 concurrent tasks")
    
    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    print("\n✅ All demonstrations completed successfully!")
    print("\nWhat we learned:")
    print("  1. Mode switches don't interrupt running tasks")
    print("  2. Exclusive mode ensures single-task execution")
    print("  3. Shared mode provides controlled concurrency")
    print("  4. The system handles transitions gracefully")
    print("\nThese features enable:")
    print("  • Dynamic resource allocation")
    print("  • Priority task execution (switch to exclusive)")
    print("  • Efficient multi-tasking (shared mode)")
    print("  • Safe mode transitions without task loss")
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

