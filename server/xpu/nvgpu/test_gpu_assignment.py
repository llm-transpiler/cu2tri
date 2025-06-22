#!/usr/bin/env python3
"""
Test script for GPU assignment functionality

This script tests the GPU ID specification feature of the GPU management API.
"""

import asyncio
import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8081"

def submit_task(task_data: Dict[str, Any]) -> str:
    """Submit a task and return task ID"""
    response = requests.post(f"{API_BASE_URL}/tasks/submit", json=task_data)
    if response.status_code == 200:
        result = response.json()
        return result["task_id"]
    else:
        raise Exception(f"Failed to submit task: {response.text}")

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status"""
    response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get task status: {response.text}")

def wait_for_task_completion(task_id: str, timeout: int = 60) -> Dict[str, Any]:
    """Wait for task to complete"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_task_status(task_id)
        if status["status"] in ["completed", "failed", "cancelled"]:
            return status
        time.sleep(2)
    raise Exception(f"Task {task_id} did not complete within {timeout} seconds")

def test_gpu_assignment():
    """Test GPU assignment functionality"""
    print("🧪 Testing GPU Assignment Functionality")
    print("=" * 50)
    
    # Test 1: Task without GPU preference
    print("\n📋 Test 1: Task without GPU preference")
    task1_data = {
        "task_type": "functional",
        "name": "No GPU Preference",
        "description": "Task without specific GPU preference",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "check_gpu_assignment",
        "args": [],
        "kwargs": {}
    }
    
    task1_id = submit_task(task1_data)
    print(f"✅ Submitted task: {task1_id}")
    
    # Test 2: Task with GPU 0 preference
    print("\n📋 Test 2: Task preferring GPU 0")
    task2_data = {
        "task_type": "functional",
        "name": "Prefer GPU 0",
        "description": "Task preferring GPU 0 with fallback",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "check_gpu_assignment",
        "args": [],
        "kwargs": {},
        "preferred_gpu_id": 0,
        "allow_fallback": True
    }
    
    task2_id = submit_task(task2_data)
    print(f"✅ Submitted task: {task2_id}")
    
    # Test 3: Task with GPU 1 preference, no fallback
    print("\n📋 Test 3: Task requiring GPU 1 (no fallback)")
    task3_data = {
        "task_type": "performance",
        "name": "Require GPU 1",
        "description": "Task requiring GPU 1 exclusively",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "check_gpu_assignment",
        "args": [],
        "kwargs": {},
        "preferred_gpu_id": 1,
        "allow_fallback": False
    }
    
    task3_id = submit_task(task3_data)
    print(f"✅ Submitted task: {task3_id}")
    
    # Test 4: GPU computation task with preference
    print("\n📋 Test 4: GPU computation task preferring GPU 2")
    task4_data = {
        "task_type": "performance",
        "name": "GPU Compute on GPU 2",
        "description": "GPU computation preferring GPU 2",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "gpu_compute_task",
        "args": [100],  # Smaller matrix for faster execution
        "kwargs": {},
        "preferred_gpu_id": 2,
        "allow_fallback": True
    }
    
    task4_id = submit_task(task4_data)
    print(f"✅ Submitted task: {task4_id}")
    
    # Test 5: GPU type matching test
    print("\n📋 Test 5: GPU type matching (prefer H100, same type only)")
    task5_data = {
        "task_type": "functional",
        "name": "H100 Type Matching",
        "description": "Task preferring H100 with same type requirement",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "check_gpu_assignment",
        "args": [],
        "kwargs": {},
        "preferred_gpu_id": 0,  # Assume GPU 0 is H100
        "allow_fallback": True,
        "require_same_gpu_type": True
    }
    
    task5_id = submit_task(task5_data)
    print(f"✅ Submitted task: {task5_id}")
    
    # Test 6: Cross-type fallback test
    print("\n📋 Test 6: Cross-type fallback (prefer H100, allow any type)")
    task6_data = {
        "task_type": "functional",
        "name": "Cross-Type Fallback",
        "description": "Task allowing fallback to different GPU types",
        "module_path": "server.xpu.nvgpu.example_tasks",
        "function_name": "check_gpu_assignment",
        "args": [],
        "kwargs": {},
        "preferred_gpu_id": 5,  # Assume GPU 0 is H100
        "allow_fallback": True,
        "require_same_gpu_type": False
    }
    
    task6_id = submit_task(task6_data)
    print(f"✅ Submitted task: {task6_id}")
    
    # Wait for all tasks to complete
    tasks = [
        (task1_id, "No GPU Preference"),
        (task2_id, "Prefer GPU 0"),
        (task3_id, "Require GPU 1"),
        (task4_id, "GPU Compute on GPU 2"),
        (task5_id, "H100 Type Matching"),
        (task6_id, "Cross-Type Fallback")
    ]
    
    print("\n⏳ Waiting for tasks to complete...")
    results = []
    
    for task_id, task_name in tasks:
        try:
            print(f"\n📊 Waiting for '{task_name}' ({task_id})...")
            status = wait_for_task_completion(task_id, timeout=120)
            results.append((task_name, status))
            
            if status["status"] == "completed":
                print(f"✅ Task completed successfully")
                print(f"   Requested GPU: {status.get('preferred_gpu_id', 'Any')}")
                print(f"   Actual GPU: {status.get('gpu_id', 'Unknown')}")
                print(f"   Allow Fallback: {status.get('allow_fallback', 'Unknown')}")
                print(f"   Require Same Type: {status.get('require_same_gpu_type', 'Unknown')}")
                if status.get("result"):
                    result = status["result"]
                    if isinstance(result, dict):
                        print(f"   CUDA_VISIBLE_DEVICES: {result.get('cuda_visible_devices', 'Unknown')}")
                        print(f"   Current Device: {result.get('current_device_id', 'Unknown')}")
                        print(f"   Device Name: {result.get('device_name', 'Unknown')}")
                        print(f"   GPU Type: {result.get('gpu_type', 'Unknown')}")
            else:
                print(f"❌ Task failed: {status.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error waiting for task: {e}")
            results.append((task_name, {"status": "error", "error": str(e)}))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    for task_name, status in results:
        status_emoji = "✅" if status["status"] == "completed" else "❌"
        print(f"{status_emoji} {task_name}: {status['status']}")
        if status["status"] == "completed" and status.get("gpu_id") is not None:
            print(f"   → Ran on GPU {status['gpu_id']}")

def check_system_status():
    """Check system status before running tests"""
    print("🔍 Checking System Status")
    print("=" * 30)
    
    try:
        # Check health
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API Server is healthy")
        else:
            print(f"❌ API Server health check failed: {response.status_code}")
            return False
        
        # Check GPU status
        response = requests.get(f"{API_BASE_URL}/gpus")
        if response.status_code == 200:
            gpus = response.json()
            print(f"✅ Found {len(gpus)} GPUs:")
            for gpu_id, gpu_info in gpus.items():
                print(f"   GPU {gpu_id}: {gpu_info['name']} "
                      f"(Memory: {gpu_info['memory_usage_ratio']:.1%}, "
                      f"Available: {gpu_info['available_for_functional']})")
        else:
            print(f"❌ Failed to get GPU status: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPU Assignment Test Suite")
    print("=" * 50)
    
    # Check system status first
    if not check_system_status():
        print("\n❌ System check failed. Make sure the API server is running.")
        exit(1)
    
    # Run tests
    try:
        test_gpu_assignment()
        print("\n🎉 All tests completed!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        exit(1) 