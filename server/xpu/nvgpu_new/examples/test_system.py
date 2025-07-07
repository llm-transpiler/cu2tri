#!/usr/bin/env python3
"""
System Test Script for GPU Management System - New Architecture

Tests the basic functionality of the new GPU management system.
"""

import asyncio
import sys
import subprocess
import time
import requests
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from server.xpu.nvgpu_new.gpu_info import query_gpu_info
from server.xpu.nvgpu_new.task import Task, TaskType, TaskStatus
from server.xpu.nvgpu_new.gpu_task_queue import GPUTaskQueue
from server.xpu.nvgpu_new.gpu_manager import GPUManager
from server.xpu.nvgpu_new.task_dispatcher import TaskDispatcher


class SystemTester:
    """Test suite for the GPU Management System"""
    
    def __init__(self):
        self.test_results = []
        self.server_process = None
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        full_message = f"[{status}] {test_name}"
        if message:
            full_message += f": {message}"
        
        print(full_message)
        self.test_results.append((test_name, passed, message))
        
        if not passed:
            print(f"  Failed: {message}")
    
    def test_gpu_info(self):
        """Test GPU information query"""
        try:
            gpu_infos = query_gpu_info(available_gpu_ids=[])
            self.log_test("GPU Info Query", 
                         len(gpu_infos) > 0, 
                         f"Found {len(gpu_infos)} GPUs")
            
            for gpu in gpu_infos:
                print(f"  GPU-{gpu.device_id}: {gpu.name} ({gpu.gpu_type.value})")
                print(f"    Memory: {gpu.memory_used_mb}/{gpu.memory_total_mb} MB")
                print(f"    Utilization: {gpu.utilization_percent}%")
                
        except Exception as e:
            self.log_test("GPU Info Query", False, str(e))
    
    async def test_task_queue(self):
        """Test GPU task queue functionality"""
        try:
            # Create a test task queue
            queue = GPUTaskQueue(gpu_id=1, max_concurrent_task_num=2)
            await queue.start()
            
            # Test task creation
            async def dummy_task():
                await asyncio.sleep(0.1)
                return {"result": "success"}
            
            task = Task(
                name="Test Task",
                description="Test task for queue",
                task_type=TaskType.SHARED,
                execute_func=dummy_task
            )
            
            # Test task submission
            task_id = await queue.submit_task(task)
            self.log_test("Task Queue Submission", 
                         task_id is not None, 
                         f"Task ID: {task_id}")
            
            # Test task status
            status = await queue.get_task_status(task_id)
            self.log_test("Task Status Query", 
                         status is not None, 
                         f"Status: {status.get('status', 'Unknown')}")
            
            # Wait a moment for task to complete
            await asyncio.sleep(0.5)
            
            # Check final status
            final_status = await queue.get_task_status(task_id)
            self.log_test("Task Completion", 
                         final_status.get('status') == 'completed',
                         f"Final status: {final_status.get('status')}")
            
            await queue.stop()
            
        except Exception as e:
            self.log_test("Task Queue Test", False, str(e))
    
    async def test_task_dispatcher(self):
        """Test task dispatcher functionality"""
        try:
            # Create dispatcher with all available GPUs
            dispatcher = TaskDispatcher(available_gpu_ids=[])
            await dispatcher.start()
            
            # Test system status
            status = await dispatcher.get_system_status()
            self.log_test("Dispatcher System Status", 
                         status.get('dispatcher_running', False),
                         f"GPUs managed: {status.get('total_gpus', 0)}")
            
            # Test task submission if GPUs are available
            if status.get('total_gpus', 0) > 0:
                async def test_task():
                    await asyncio.sleep(0.1)
                    return {"test": "success"}
                
                task_id = await dispatcher.submit_task(
                    task_type=TaskType.SHARED,
                    name="Dispatcher Test",
                    description="Test task for dispatcher",
                    execute_func=test_task
                )
                
                self.log_test("Dispatcher Task Submission", 
                             task_id is not None,
                             f"Task ID: {task_id}")
                
                # Wait for task completion
                await asyncio.sleep(0.5)
                
                task_status = await dispatcher.get_task_status(task_id)
                self.log_test("Dispatcher Task Execution",
                             task_status.get('status') == 'completed',
                             f"Status: {task_status.get('status')}")
            else:
                self.log_test("Dispatcher Task Test", False, "No GPUs available")
            
            await dispatcher.stop()
            
        except Exception as e:
            self.log_test("Task Dispatcher Test", False, str(e))
    
    def start_api_server(self, port=8080):
        """Start the API server for testing"""
        try:
            # Start server in background
            cmd = [
                sys.executable, 
                "start_server.py", 
                "--port", str(port),
                "--log-level", "WARNING"  # Reduce log noise
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    self.log_test("API Server Start", True, f"Server running on port {port}")
                    return True
                else:
                    self.log_test("API Server Start", False, f"Server returned {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                self.log_test("API Server Start", False, f"Connection failed: {e}")
                return False
                
        except Exception as e:
            self.log_test("API Server Start", False, str(e))
            return False
    
    def stop_api_server(self):
        """Stop the API server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
            self.server_process = None
    
    def test_api_endpoints(self, port=8080):
        """Test API endpoints"""
        base_url = f"http://localhost:{port}"
        
        try:
            # Test root endpoint
            response = requests.get(f"{base_url}/")
            self.log_test("API Root Endpoint", 
                         response.status_code == 200,
                         f"Response: {response.status_code}")
            
            # Test status endpoint
            response = requests.get(f"{base_url}/status")
            self.log_test("API Status Endpoint", 
                         response.status_code == 200,
                         f"Response: {response.status_code}")
            
            if response.status_code == 200:
                status_data = response.json()
                print(f"  System running: {status_data.get('dispatcher_running', False)}")
                print(f"  Total GPUs: {status_data.get('total_gpus', 0)}")
            
            # Test GPU endpoint
            response = requests.get(f"{base_url}/gpus/available")
            self.log_test("API GPU Endpoint", 
                         response.status_code == 200,
                         f"Response: {response.status_code}")
            
            # Test task types endpoint
            response = requests.get(f"{base_url}/task-types")
            self.log_test("API Task Types Endpoint", 
                         response.status_code == 200,
                         f"Response: {response.status_code}")
            
            # Test task submission (if GPUs available)
            gpu_response = requests.get(f"{base_url}/gpus/available")
            if gpu_response.status_code == 200:
                gpu_data = gpu_response.json()
                if gpu_data.get('total_gpus', 0) > 0:
                    self.test_task_submission_api(base_url)
                else:
                    self.log_test("API Task Submission", False, "No GPUs available")
            
        except Exception as e:
            self.log_test("API Endpoints Test", False, str(e))
    
    def test_task_submission_api(self, base_url: str):
        """Test task submission via API"""
        try:
            # Submit a simple shared task
            task_payload = {
                "task_type": "shared",
                "name": "API Test Task",
                "description": "Test task submitted via API",
                "module_path": "server.xpu.nvgpu_new.examples.simple_tasks",
                "function_name": "simple_shared_task",
                "kwargs": {"duration": 1.0, "use_gpu": False}  # Use CPU to avoid GPU dependency
            }
            
            response = requests.post(f"{base_url}/tasks/submit", json=task_payload)
            self.log_test("API Task Submission", 
                         response.status_code == 200,
                         f"Response: {response.status_code}")
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data.get('task_id')
                
                if task_id:
                    # Wait for task completion
                    time.sleep(3)
                    
                    # Check task status
                    status_response = requests.get(f"{base_url}/tasks/{task_id}")
                    self.log_test("API Task Status Check", 
                                 status_response.status_code == 200,
                                 f"Status response: {status_response.status_code}")
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        task_status = status_data.get('status')
                        self.log_test("API Task Execution", 
                                     task_status in ['completed', 'running'],
                                     f"Task status: {task_status}")
            
        except Exception as e:
            self.log_test("API Task Submission", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("GPU Management System - Test Suite")
        print("==================================")
        
        # Basic functionality tests
        print("\n1. Testing GPU Information Query...")
        self.test_gpu_info()
        
        print("\n2. Testing Task Queue...")
        await self.test_task_queue()
        
        print("\n3. Testing Task Dispatcher...")
        await self.test_task_dispatcher()
        
        # API tests
        print("\n4. Testing API Server...")
        server_started = self.start_api_server()
        
        if server_started:
            print("\n5. Testing API Endpoints...")
            self.test_api_endpoints()
            
            print("\n6. Stopping API Server...")
            self.stop_api_server()
            self.log_test("API Server Stop", True, "Server stopped")
        
        # Test summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_name, passed, message in self.test_results:
                if not passed:
                    print(f"  - {test_name}: {message}")
        
        print(f"\nOverall Status: {'PASS' if failed_tests == 0 else 'FAIL'}")
        return failed_tests == 0


async def main():
    """Main test execution"""
    tester = SystemTester()
    
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        sys.exit(1)
    finally:
        # Ensure server is stopped
        tester.stop_api_server()


if __name__ == "__main__":
    asyncio.run(main()) 