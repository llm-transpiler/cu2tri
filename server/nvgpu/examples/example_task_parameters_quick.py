#!/usr/bin/env python3
"""
Quick test: Essential task parameter combinations

Tests the most important combinations to ensure robustness.
"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from client import NVGPUClient
import time

def main():
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("❌ Server is not running!")
        return 1
    
    print("\n" + "="*70)
    print("  Quick Test: Task Parameters (type, mode, label)")
    print("="*70 + "\n")
    
    script = "/workspace/server/nvgpu/test_scripts/simple_functional_test.py"
    
    test_cases = [
        # (Description, kwargs, expected_mode)
        ("No parameters", {}, "shared"),
        ("Only mode=shared", {"task_mode": "shared"}, "shared"),
        ("Only mode=exclusive", {"task_mode": "exclusive"}, "exclusive"),
        ("Only type=functional", {"task_type": "functional"}, "shared"),
        ("Only type=performance", {"task_type": "performance"}, "exclusive"),
        ("Only type=both", {"task_type": "both"}, "exclusive"),
        ("Only label", {"task_label": "test"}, "shared"),
        ("type=functional + label", {"task_type": "functional", "task_label": "t1"}, "shared"),
        ("type=performance + label", {"task_type": "performance", "task_label": "t2"}, "exclusive"),
        ("Override: type=functional, mode=exclusive", {"task_type": "functional", "task_mode": "exclusive"}, "exclusive"),
        ("Override: type=performance, mode=shared", {"task_type": "performance", "task_mode": "shared"}, "shared"),
        ("All params", {"task_type": "functional", "task_mode": "shared", "task_label": "all"}, "shared"),
    ]
    
    passed = 0
    failed = 0
    
    for desc, kwargs, expected_mode in test_cases:
        try:
            task_id = client.submit_task(script_path=script, **kwargs)
            time.sleep(0.3)
            
            result = client.get_task(task_id)
            actual_mode = result.task_mode if hasattr(result, 'task_mode') else None
            
            # Wait for completion
            try:
                final = client.wait_for_task(task_id, timeout=30)
                completed = final.status == "completed" and final.exit_code == 0
            except:
                completed = False
            
            match = actual_mode == expected_mode
            
            if completed and match:
                icon = "✓"
                passed += 1
            else:
                icon = "✗"
                failed += 1
            
            # Format parameters
            if kwargs:
                params = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            else:
                params = "none"
            
            print(f"{icon} {desc}")
            print(f"  Params: {params}")
            print(f"  Expected: {expected_mode}, Got: {actual_mode}, Status: {final.status if 'final' in locals() else 'unknown'}")
            
            if not match or not completed:
                print(f"  ⚠️  Issue detected!")
            print()
            
        except Exception as e:
            failed += 1
            print(f"✗ {desc}")
            print(f"  ERROR: {e}")
            print()
    
    print("="*70)
    print(f"\nResults: {passed}/{passed+failed} passed")
    
    if failed == 0:
        print("🎉 All tests passed!\n")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

