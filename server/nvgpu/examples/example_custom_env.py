#!/usr/bin/env python3
"""Custom environment example: pass custom environment variables to tasks."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import NVGPUClient


def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("ERROR: Server not responding")
        return 1
    
    print("=== Custom Environment Example ===\n")
    
    # Create a simple test script that uses environment variables
    test_script = "/tmp/env_test.py"
    with open(test_script, "w") as f:
        f.write("""#!/usr/bin/env python3
import os
import sys

print("Environment variables:")
print(f"  MY_VAR: {os.environ.get('MY_VAR', 'not set')}")
print(f"  TEST_MODE: {os.environ.get('TEST_MODE', 'not set')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Verify our custom env vars are set
my_var = os.environ.get('MY_VAR')
test_mode = os.environ.get('TEST_MODE')

if my_var == "hello" and test_mode == "debug":
    print("\\n✓ Custom environment variables set correctly")
    sys.exit(0)
else:
    print("\\n✗ Custom environment variables not set correctly")
    sys.exit(1)
""")
    
    os.chmod(test_script, 0o755)
    
    # Submit task with custom environment
    print("Submitting task with custom environment variables...")
    task_id = client.submit_task(
        script_path=test_script,
        task_type="functional",
        env={
            "MY_VAR": "hello",
            "TEST_MODE": "debug"
        }
    )
    print(f"Task ID: {task_id}\n")
    
    # Wait for completion
    print("Waiting for task to complete...")
    result = client.wait_for_task(task_id, timeout=60)
    
    print(f"\nTask status: {result.status}")
    print(f"Exit code: {result.exit_code}")
    
    # Read and display log file
    if result.log_file and os.path.exists(result.log_file):
        print(f"\n=== Task Log ===")
        with open(result.log_file, "r") as f:
            content = f.read()
            # Show only stdout section
            if "=== STDOUT ===" in content:
                stdout = content.split("=== STDOUT ===")[1].split("=== STDERR ===")[0]
                print(stdout.strip())
    
    # Clean up
    os.remove(test_script)
    
    if result.status == "completed" and result.exit_code == 0:
        print("\n✓ Test PASSED")
        return 0
    else:
        print("\n✗ Test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

