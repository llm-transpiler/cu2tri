#!/usr/bin/env python3
"""
Example: Test task parameters (task_type, task_mode, task_label)

Demonstrates that tasks work correctly with various combinations of:
- task_type (functional, performance, both)
- task_mode (exclusive, shared)
- task_label (custom string)

Tests smart defaults and parameter combinations.
"""

import sys
sys.path.insert(0, '/workspace/server/nvgpu')

from client import NVGPUClient
import time

def test_task_parameters():
    """Test various combinations of task parameters."""
    client = NVGPUClient("http://localhost:8080")
    
    if not client.health_check():
        print("❌ Server is not running!")
        return 1
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         Task Parameters Test (type, mode, label)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Test cases: (name, kwargs, expected_mode)
    test_cases = [
        # 1. Only task_mode (explicit)
        (
            "1. Only task_mode='shared'",
            {"task_mode": "shared"},
            "shared"
        ),
        (
            "2. Only task_mode='exclusive'",
            {"task_mode": "exclusive"},
            "exclusive"
        ),
        
        # 3-5. Only task_type (smart defaults)
        (
            "3. Only task_type='functional' (default: shared)",
            {"task_type": "functional"},
            "shared"
        ),
        (
            "4. Only task_type='performance' (default: exclusive)",
            {"task_type": "performance"},
            "exclusive"
        ),
        (
            "5. Only task_type='both' (default: exclusive)",
            {"task_type": "both"},
            "exclusive"
        ),
        
        # 6. Only task_label (no smart defaults)
        (
            "6. Only task_label (default: shared)",
            {"task_label": "custom_test_label"},
            "shared"
        ),
        
        # 7-8. task_type + task_label
        (
            "7. task_type='functional' + task_label",
            {"task_type": "functional", "task_label": "func_test_1"},
            "shared"
        ),
        (
            "8. task_type='performance' + task_label",
            {"task_type": "performance", "task_label": "perf_test_1"},
            "exclusive"
        ),
        
        # 9-10. task_mode overrides smart defaults
        (
            "9. task_type='functional' + task_mode='exclusive' (override)",
            {"task_type": "functional", "task_mode": "exclusive"},
            "exclusive"
        ),
        (
            "10. task_type='performance' + task_mode='shared' (override)",
            {"task_type": "performance", "task_mode": "shared"},
            "shared"
        ),
        
        # 11-12. All three parameters
        (
            "11. All params: type='functional', mode='shared', label='test'",
            {"task_type": "functional", "task_mode": "shared", "task_label": "all_params_1"},
            "shared"
        ),
        (
            "12. All params: type='performance', mode='exclusive', label='test'",
            {"task_type": "performance", "task_mode": "exclusive", "task_label": "all_params_2"},
            "exclusive"
        ),
        
        # 13. No parameters at all (full defaults)
        (
            "13. No parameters (default: shared)",
            {},
            "shared"
        ),
        
        # 14-15. Edge cases
        (
            "14. task_type='both' + task_label",
            {"task_type": "both", "task_label": "both_test"},
            "exclusive"
        ),
        (
            "15. task_mode='exclusive' + task_label (no type)",
            {"task_mode": "exclusive", "task_label": "exclusive_only"},
            "exclusive"
        ),
    ]
    
    results = []
    script_path = "/workspace/server/nvgpu/test_scripts/simple_functional_test.py"
    
    print("Running tests...\n")
    print("─" * 70)
    
    for i, (name, kwargs, expected_mode) in enumerate(test_cases, 1):
        try:
            # Submit task
            task_id = client.submit_task(
                script_path=script_path,
                **kwargs
            )
            
            # Wait a bit for task to be processed
            time.sleep(0.5)
            
            # Get task details
            result = client.get_task(task_id)
            
            # Check if mode matches expected
            actual_mode = result.task_mode if hasattr(result, 'task_mode') else "unknown"
            mode_match = actual_mode == expected_mode
            
            # Wait for completion (with timeout)
            try:
                final_result = client.wait_for_task(task_id, timeout=30)
                success = final_result.status == "completed" and final_result.exit_code == 0
            except TimeoutError:
                success = False
                final_result = client.get_task(task_id)
            
            status_icon = "✓" if success and mode_match else "✗"
            mode_icon = "✓" if mode_match else "✗"
            
            results.append({
                "name": name,
                "success": success and mode_match,
                "mode_match": mode_match,
                "actual_mode": actual_mode,
                "expected_mode": expected_mode,
                "status": final_result.status,
                "task_id": task_id[:8]
            })
            
            # Print result
            print(f"{status_icon} {name}")
            print(f"   Task ID: {task_id[:8]}...")
            print(f"   Expected mode: {expected_mode}")
            print(f"   Actual mode:   {actual_mode} {mode_icon}")
            print(f"   Status: {final_result.status}")
            
            # Show parameters used
            params_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            if not params_str:
                params_str = "(no parameters)"
            print(f"   Parameters: {params_str}")
            print()
            
        except Exception as e:
            print(f"✗ {name}")
            print(f"   ERROR: {str(e)}")
            print()
            results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
    
    print("─" * 70)
    print("\n📊 Summary\n")
    
    passed = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    print(f"Total: {total} tests")
    print(f"Passed: {passed} tests ✓")
    print(f"Failed: {total - passed} tests ✗")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return_code = 0
    else:
        print("\n⚠️  Some tests failed:")
        for r in results:
            if not r.get("success", False):
                print(f"  - {r['name']}")
                if "error" in r:
                    print(f"    Error: {r['error']}")
                elif not r.get("mode_match", True):
                    print(f"    Expected mode: {r['expected_mode']}, got: {r['actual_mode']}")
        return_code = 1
    
    # Smart defaults verification
    print("\n" + "─" * 70)
    print("\n📋 Smart Defaults Verification\n")
    
    smart_default_tests = [
        (3, "functional", "shared"),
        (4, "performance", "exclusive"),
        (5, "both", "exclusive"),
    ]
    
    print("Task type → Default mode:")
    for test_num, task_type, expected_mode in smart_default_tests:
        result = results[test_num - 1]
        match = "✓" if result.get("mode_match", False) else "✗"
        print(f"  {match} {task_type:12} → {expected_mode}")
    
    print("\n" + "─" * 70)
    print("\n📋 Override Verification\n")
    
    print("Can task_mode override smart defaults?")
    override_tests = [
        (9, "functional + exclusive override", "exclusive"),
        (10, "performance + shared override", "shared"),
    ]
    
    for test_num, desc, expected_mode in override_tests:
        result = results[test_num - 1]
        match = "✓" if result.get("mode_match", False) else "✗"
        print(f"  {match} {desc:35} → {expected_mode}")
    
    print("\n" + "╚" + "═" * 68 + "╝")
    
    return return_code


if __name__ == "__main__":
    sys.exit(test_task_parameters())

