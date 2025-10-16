#!/usr/bin/env python3
"""
Example: Task Timing Analysis

This example demonstrates how to use the new timing fields to analyze task performance.

New timing fields (all in milliseconds with 2 decimal places):
- pending_time_ms: Time spent waiting for GPU assignment
- queue_time_ms: Time spent in GPU queue waiting for execution
- waiting_time_ms: Total waiting time (pending + queue)
- running_time_ms: Actual execution duration
- total_time_ms: Total time from submit to completion

Timeline:
  submit → queued → start → end
      |      |        |      |
      |<-pending->|   |      |
      |      |<-queue->|      |
      |<--waiting-->|        |
      |              |<-run->|
      |<-----total----------->|
"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from client import NVGPUClient


def print_timing_breakdown(task_id: str, result):
    """Print detailed timing breakdown."""
    print(f"\n{'='*70}")
    print(f"Task: {task_id[:12]}...")
    print(f"Status: {result.status} (exit_code={result.exit_code})")
    print(f"{'='*70}")
    
    if result.queued_timestamp:
        print(f"\n📅 Timestamps:")
        print(f"  Submit Timestamp:  {result.submit_timestamp}")
        print(f"  Queued Timestamp:  {result.queued_timestamp}")
        print(f"  Start Timestamp:   {result.start_timestamp}")
        print(f"  End Timestamp:     {result.end_timestamp}")
    
    if result.total_time_ms is not None:
        print(f"\n⏱️  Timing Breakdown (milliseconds):")
        print(f"  ┌─────────────────────────────┬──────────────┬──────────┐")
        print(f"  │          Phase              │   Time (ms)  │  % Total │")
        print(f"  ├─────────────────────────────┼──────────────┼──────────┤")
        
        if result.pending_time_ms is not None:
            pct = (result.pending_time_ms / result.total_time_ms * 100) if result.total_time_ms > 0 else 0
            print(f"  │  Pending (→GPU assign)      │ {result.pending_time_ms:>10.2f}   │ {pct:>6.2f}%  │")
        
        if result.queue_time_ms is not None:
            pct = (result.queue_time_ms / result.total_time_ms * 100) if result.total_time_ms > 0 else 0
            print(f"  │  Queue (→execution start)   │ {result.queue_time_ms:>10.2f}   │ {pct:>6.2f}%  │")
        
        if result.waiting_time_ms is not None:
            pct = (result.waiting_time_ms / result.total_time_ms * 100) if result.total_time_ms > 0 else 0
            print(f"  │  Total Waiting              │ {result.waiting_time_ms:>10.2f}   │ {pct:>6.2f}%  │")
        
        running_ms = result.running_time_ms
        if running_ms is not None:
            pct = (running_ms / result.total_time_ms * 100) if result.total_time_ms > 0 else 0
            print(f"  │  Running (actual runtime)   │ {running_ms:>10.2f}   │ {pct:>6.2f}%  │")
        
        print(f"  ├─────────────────────────────┼──────────────┼──────────┤")
        print(f"  │  TOTAL TIME                 │ {result.total_time_ms:>10.2f}   │ 100.00%  │")
        print(f"  └─────────────────────────────┴──────────────┴──────────┘")
        
        # Performance insights
        if result.waiting_time_ms and running_ms:
            wait_ratio = result.waiting_time_ms / running_ms
            print(f"\n📊 Performance Insights:")
            print(f"  • Wait-to-Execution Ratio: {wait_ratio:.2f}x")
            if wait_ratio > 2:
                print(f"  ⚠️  High waiting time! Task spent {wait_ratio:.1f}x more time waiting than executing.")
            elif wait_ratio > 0.5:
                print(f"  ℹ️  Moderate waiting time.")
            else:
                print(f"  ✓ Low waiting time, good throughput!")
            
            if result.pending_time_ms and result.queue_time_ms:
                if result.pending_time_ms > result.queue_time_ms:
                    print(f"  • Bottleneck: GPU assignment (pending phase)")
                else:
                    print(f"  • Bottleneck: GPU queue (waiting for GPU availability)")


def main():
    client = NVGPUClient("http://localhost:8080")
    
    print("="*70)
    print("Task Timing Analysis Example")
    print("="*70)
    
    # Submit multiple tasks to demonstrate timing
    print("\n1️⃣  Submitting tasks...")
    
    task_ids = []
    for i in range(3):
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional",
            task_label=f"timing_test_{i+1}"
        )
        task_ids.append(task_id)
        print(f"  ✓ Submitted task {i+1}: {task_id[:12]}...")
    
    # Wait for all tasks to complete
    print(f"\n2️⃣  Waiting for tasks to complete...")
    for i, task_id in enumerate(task_ids):
        print(f"  ⏳ Waiting for task {i+1}...")
        result = client.wait_for_task(task_id, timeout=60)
        print(f"  ✓ Task {i+1} {result.status}")
    
    # Display timing analysis for each task
    print(f"\n3️⃣  Timing Analysis:")
    for i, task_id in enumerate(task_ids):
        result = client.get_task(task_id)
        print_timing_breakdown(task_id, result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    total_execution = 0
    total_waiting = 0
    for task_id in task_ids:
        result = client.get_task(task_id)
        running_ms = result.running_time_ms
        if running_ms:
            total_execution += running_ms
        if result.waiting_time_ms:
            total_waiting += result.waiting_time_ms
    
    print(f"Total Execution Time: {total_execution:.2f} ms")
    print(f"Total Waiting Time:   {total_waiting:.2f} ms")
    print(f"Total Time:           {total_execution + total_waiting:.2f} ms")
    
    if total_execution > 0:
        efficiency = (total_execution / (total_execution + total_waiting)) * 100
        print(f"Execution Efficiency: {efficiency:.1f}%")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
