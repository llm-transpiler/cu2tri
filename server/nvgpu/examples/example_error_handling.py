#!/usr/bin/env python3
"""Error handling example: handle task failures gracefully."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("ERROR: Server not responding")
        return 1
    
    print("=== Error Handling Example ===\n")
    
    # Test 1: Successful task
    print("Test 1: Submitting successful task...")
    try:
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional"
        )
        result = client.wait_for_task(task_id, timeout=60)
        
        if result.status == "completed" and result.exit_code == 0:
            print("✓ Task succeeded as expected\n")
        else:
            print(f"✗ Task failed unexpectedly: {result.error_message}\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test 2: Task with exception
    print("Test 2: Submitting task that raises exception...")
    try:
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/failing_test.py",
            task_type="functional",
            args=["--error-type", "exception"]
        )
        result = client.wait_for_task(task_id, timeout=60)
        
        if result.status == "failed":
            print(f"✓ Task failed as expected: {result.error_message}")
            print(f"  Exit code: {result.exit_code}")
            print(f"  Log file: {result.log_file}\n")
        else:
            print(f"✗ Task should have failed but status is: {result.status}\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test 3: Task with non-zero exit code
    print("Test 3: Submitting task with non-zero exit code...")
    try:
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/failing_test.py",
            task_type="functional",
            args=["--error-code", "42"]
        )
        result = client.wait_for_task(task_id, timeout=60)
        
        if result.status == "failed" and result.exit_code == 42:
            print(f"✓ Task failed with correct exit code: {result.exit_code}\n")
        else:
            print(f"✗ Unexpected result: status={result.status}, exit_code={result.exit_code}\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test 4: Cancel a running task (using force)
    print("Test 4: Submitting and force-cancelling a task...")
    try:
        # Submit a long-running task
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "300"]
        )
        print(f"  Task submitted: {task_id}")
        
        # Wait for it to start running
        import time
        time.sleep(2)
        
        # Force cancel (even if running)
        if client.cancel_task(task_id, force=True):
            print("  ✓ Task force-cancelled successfully")
            
            # Verify cancellation
            time.sleep(1)
            result = client.get_task(task_id)
            print(f"  Task status: {result.status}\n")
        else:
            print("  ✗ Failed to force cancel task\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test 5: Handle timeout
    print("Test 5: Testing timeout handling...")
    try:
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "60"]
        )
        
        # Try to wait with a short timeout
        result = client.wait_for_task(task_id, timeout=5)
        print(f"✗ Should have timed out but got: {result.status}\n")
    
    except TimeoutError as e:
        print(f"✓ Timed out as expected: {e}")
        # Force cancel the running task
        if client.cancel_task(task_id, force=True):
            print("  Task force-cancelled\n")
        else:
            print("  Warning: Failed to cancel task\n")
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")
    
    print("=== Error Handling Tests Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

