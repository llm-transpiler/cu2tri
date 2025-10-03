#!/usr/bin/env python3
"""GPU management example: manage GPU settings dynamically."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("ERROR: Server not responding")
        return 1
    
    print("=== GPU Management Example ===\n")
    
    # List all GPUs
    print("Current GPUs:")
    gpus = client.list_gpus()
    for gpu in gpus:
        print(f"  GPU {gpu['gpu_id']}:")
        print(f"    Mode: {gpu['mode']}")
        print(f"    Status: {gpu['status']}")
        print(f"    Memory threshold: {gpu['memory_threshold']*100:.0f}%")
        print(f"    Current memory: {gpu['current_memory_usage']*100:.1f}%")
        print(f"    Running tasks: {len(gpu['running_tasks'])}")
    
    if not gpus:
        print("  No GPUs registered")
        return 0
    
    gpu_id = gpus[0]['gpu_id']
    print(f"\n=== Managing GPU {gpu_id} ===\n")
    
    # Change to exclusive mode
    print("1. Setting GPU to exclusive mode...")
    if client.set_gpu_mode(gpu_id, "exclusive"):
        print("   ✓ Success")
    else:
        print("   ✗ Failed")
    
    # Check new settings
    gpu = client.get_gpu(gpu_id)
    print(f"   Current mode: {gpu['mode']}")
    
    # Change memory threshold
    print("\n2. Setting memory threshold to 80%...")
    if client.set_gpu_memory_threshold(gpu_id, 0.8):
        print("   ✓ Success")
    else:
        print("   ✗ Failed")
    
    # Check new settings
    gpu = client.get_gpu(gpu_id)
    print(f"   Current threshold: {gpu['memory_threshold']*100:.0f}%")
    
    # Submit a task on this GPU
    print(f"\n3. Submitting task to GPU {gpu_id}...")
    task_id = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
        task_type="functional",
        gpu_id=gpu_id
    )
    print(f"   Task ID: {task_id}")
    
    # Wait for task
    print("   Waiting for completion...")
    result = client.wait_for_task(task_id, timeout=300)
    print(f"   Status: {result.status}, Exit code: {result.exit_code}")
    
    # Restore to shared mode
    print("\n4. Restoring GPU to shared mode...")
    if client.set_gpu_mode(gpu_id, "shared"):
        print("   ✓ Success")
    else:
        print("   ✗ Failed")
    
    # Get statistics
    print("\n=== Server Statistics ===")
    stats = client.get_stats()
    print(f"Queue:")
    print(f"  Pending: {stats['queue']['pending']}")
    print(f"  Queued: {stats['queue']['queued']}")
    print(f"  Running: {stats['queue']['running']}")
    print(f"  Completed: {stats['queue']['completed']}")
    print(f"  Failed: {stats['queue']['failed']}")
    print(f"\nGPUs:")
    print(f"  Total: {stats['gpus']['total_gpus']}")
    print(f"  Online: {stats['gpus']['online_gpus']}")
    print(f"  Severe error active: {stats['gpus']['severe_error_active']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

