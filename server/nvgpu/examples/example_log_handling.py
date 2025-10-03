#!/usr/bin/env python3
"""
Example: Task log handling

Demonstrates:
- Getting task log summary, stdout, and stderr
- Handling large log files with pagination
- Getting stdout/stderr sizes
"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from client import NVGPUClient
import time

def format_size(bytes_size: int) -> str:
    """Format size in human-readable format."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f}GB"

def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("Server is not running!")
        return 1
    
    print("=== Task Log Handling Example ===\n")
    
    # Submit a task
    print("1. Submitting task...")
    task_id = client.submit_task(
        script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
        task_type="functional"
    )
    print(f"   Task ID: {task_id}\n")
    
    # Wait for completion
    print("2. Waiting for task to complete...")
    result = client.wait_for_task(task_id, timeout=60)
    print(f"   Status: {result.status}")
    print(f"   Exit Code: {result.exit_code}")
    print(f"   STDOUT size: {format_size(result.stdout_size)}")
    print(f"   STDERR size: {format_size(result.stderr_size)}")
    print(f"   Log file: {result.log_file}\n")
    
    # Get log summary
    print("3. Getting log summary...")
    summary = client.get_task_log(task_id, log_type="summary")
    print(f"   Total size: {format_size(summary['total_size'])}")
    print(f"   Preview (first 500 chars):")
    print("   " + "-" * 60)
    print("   " + summary['content'][:500].replace('\n', '\n   '))
    if summary['truncated']:
        print(f"   ... (truncated, {format_size(summary['total_size'] - 500)} remaining)")
    print("   " + "-" * 60 + "\n")
    
    # Get stdout
    if result.stdout_size > 0:
        print("4. Getting STDOUT...")
        stdout = client.get_task_log(task_id, log_type="stdout", limit=1024)
        print(f"   Size: {format_size(stdout['total_size'])}")
        if stdout['total_size'] > 0:
            print(f"   Content preview:")
            print("   " + "-" * 60)
            print("   " + stdout['content'][:500].replace('\n', '\n   '))
            if stdout['has_more']:
                print(f"   ... (more data available)")
            print("   " + "-" * 60 + "\n")
    else:
        print("4. STDOUT is empty\n")
    
    # Get stderr
    if result.stderr_size > 0:
        print("5. Getting STDERR...")
        stderr = client.get_task_log(task_id, log_type="stderr", limit=1024)
        print(f"   Size: {format_size(stderr['total_size'])}")
        if stderr['total_size'] > 0:
            print(f"   Content preview:")
            print("   " + "-" * 60)
            print("   " + stderr['content'][:500].replace('\n', '\n   '))
            if stderr['has_more']:
                print(f"   ... (more data available)")
            print("   " + "-" * 60 + "\n")
    else:
        print("5. STDERR is empty\n")
    
    # Demonstrate pagination for large logs
    print("6. Demonstrating pagination...")
    print("   Reading log summary in 1KB chunks:")
    offset = 0
    chunk_size = 1024
    chunk_num = 1
    
    while True:
        chunk = client.get_task_log(task_id, log_type="summary", offset=offset, limit=chunk_size)
        print(f"   Chunk {chunk_num}: offset={offset}, size={chunk['size']}, has_more={chunk['has_more']}")
        
        if not chunk['has_more']:
            break
        
        offset += chunk['size']
        chunk_num += 1
    
    print("\n   Using get_full_task_log to fetch entire log:")
    full_log = client.get_full_task_log(task_id, log_type="summary")
    print(f"   Full log size: {format_size(len(full_log.encode('utf-8')))}\n")
    
    print("=== Example Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())

