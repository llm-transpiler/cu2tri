#!/usr/bin/env python3
"""Basic example: submit a simple task and wait for completion."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def main():
    # Create client
    client = NVGPUClient("http://localhost:8080")
    
    # Check server health
    if not client.health_check():
        print("ERROR: Server is not responding")
        return 1
    
    print("Server is healthy")
    
    # Submit a simple functional test
    print("\nSubmitting functional test...")
    task_id = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
        task_type="functional"
    )
    print(f"Task submitted: {task_id}")
    
    # Wait for completion
    print("Waiting for task to complete...")
    result = client.wait_for_task(task_id, timeout=300)
    
    # Check result
    print(f"\nTask completed with status: {result.status}")
    print(f"Exit code: {result.exit_code}")
    print(f"Log file: {result.log_file}")
    
    if result.status == "completed" and result.exit_code == 0:
        print("\n✓ Test PASSED")
        return 0
    else:
        print(f"\n✗ Test FAILED: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

