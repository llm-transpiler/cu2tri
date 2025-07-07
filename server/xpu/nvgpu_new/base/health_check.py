"""
GPU Health Check Module

Provides lightweight and isolated GPU health checking functionality.
Uses minimal CUDA operations to verify GPU availability without leaving
memory footprint.
"""

import asyncio
import logging
import os
import tempfile
from typing import Optional
import json


class GPUHealthChecker:
    """
    Lightweight GPU health checker that performs isolated GPU validation.
    
    This checker uses subprocess isolation to ensure no memory leaks or
    persistent CUDA contexts are left behind.
    """
    
    def __init__(self, gpu_id: int, logger: Optional[logging.Logger] = None):
        self.gpu_id = gpu_id
        self.logger = logger or logging.getLogger(__name__)
        self.timeout_seconds = 30
        
    async def check_gpu_health(self) -> bool:
        """
        Perform lightweight GPU health check.
        
        Returns:
            bool: True if GPU is healthy and available, False otherwise
        """
        try:
            # Run health check in isolated subprocess
            result = await self._run_isolated_health_check()
            
            if result['success']:
                self.logger.debug(f"[GPU-{self.gpu_id}] Health check passed: {result['info']}")
                return True
            else:
                self.logger.warning(f"[GPU-{self.gpu_id}] Health check failed: {result['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"[GPU-{self.gpu_id}] Health check error: {e}")
            return False
    
    async def _run_isolated_health_check(self) -> dict:
        """
        Run health check in isolated subprocess to avoid memory leaks.
        
        Returns:
            dict: Result with 'success', 'info', and 'error' keys
        """
        # Create temporary script for isolated execution
        script_content = self._generate_health_check_script()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run the script in isolated process
            cmd = [
                'python', script_path,
                '--gpu_id', str(self.gpu_id),
                '--timeout', str(self.timeout_seconds)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_clean_environment()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout_seconds
                )
                
                if process.returncode == 0:
                    # Parse JSON result
                    result = json.loads(stdout.decode().strip())
                    return result
                else:
                    error_msg = stderr.decode().strip() if stderr else "Unknown error"
                    return {
                        'success': False,
                        'error': f"Health check process failed: {error_msg}",
                        'info': None
                    }
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'error': f"Health check timed out after {self.timeout_seconds} seconds",
                    'info': None
                }
                
        finally:
            # Clean up temporary script
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _generate_health_check_script(self) -> str:
        """
        Generate the isolated health check script.
        
        Returns:
            str: Python script content for health checking
        """
        script = '''
import argparse
import json
import sys
import os
import gc
import time

def check_gpu_health(gpu_id: int, timeout: int) -> dict:
    """
    Perform minimal GPU health check with complete isolation.
    
    Args:
        gpu_id: CUDA device ID to check
        timeout: Timeout in seconds
        
    Returns:
        dict: Result with success, info, and error keys
    """
    try:
        # Set CUDA_VISIBLE_DEVICES to only the target GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Import torch after setting CUDA_VISIBLE_DEVICES
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return {
                'success': False,
                'error': 'CUDA is not available',
                'info': None
            }
        
        # Check if the GPU is accessible
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return {
                'success': False,
                'error': 'No CUDA devices found',
                'info': None
            }
        
        # Since we set CUDA_VISIBLE_DEVICES, device 0 is our target GPU
        device = torch.device('cuda:0')
        
        # Get device properties
        try:
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get device properties: {str(e)}',
                'info': None
            }
        
        # Perform minimal tensor operations to verify GPU functionality
        try:
            # Create two small tensors
            a = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)
            b = torch.tensor([3.0, 4.0], device=device, dtype=torch.float32)
            
            # Perform simple addition
            c = a + b
            
            # Verify result
            expected = torch.tensor([4.0, 6.0], device=device, dtype=torch.float32)
            if not torch.allclose(c, expected):
                return {
                    'success': False,
                    'error': 'GPU computation produced incorrect results',
                    'info': None
                }
            
            # Perform matrix multiplication to test more complex operations
            x = torch.randn(2, 2, device=device)
            y = torch.randn(2, 2, device=device)
            z = torch.mm(x, y)
            
            # Verify the operation completed without error
            if z.device != device:
                return {
                    'success': False,
                    'error': 'GPU computation returned tensor on wrong device',
                    'info': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'GPU computation failed: {str(e)}',
                'info': None
            }
        
        # Force cleanup of GPU memory
        try:
            del a, b, c, x, y, z, expected
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Additional cleanup
            gc.collect()
            
        except Exception as e:
            # Don't fail the health check for cleanup errors
            pass
        
        # Return success with device info
        return {
            'success': True,
            'error': None,
            'info': {
                'device_name': device_name,
                'compute_capability': f"{device_capability[0]}.{device_capability[1]}",
                'memory_total_gb': round(memory_total / (1024**3), 2),
                'gpu_id': gpu_id
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Health check exception: {str(e)}',
            'info': None
        }
    
    finally:
        # Force cleanup and garbage collection
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='GPU Health Check')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU device ID')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    # Perform health check
    result = check_gpu_health(args.gpu_id, args.timeout)
    
    # Output result as JSON
    print(json.dumps(result))
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == '__main__':
    main()
'''
        return script
    
    def _get_clean_environment(self) -> dict:
        """
        Get clean environment variables for subprocess.
        
        Returns:
            dict: Clean environment variables
        """
        # Start with current environment
        env = os.environ.copy()
        
        # Remove any existing CUDA_VISIBLE_DEVICES to avoid conflicts
        env.pop('CUDA_VISIBLE_DEVICES', None)
        
        # Remove any torch-related cache variables
        env.pop('TORCH_HOME', None)
        env.pop('TORCH_CACHE_DIR', None)
        
        # Set minimal CUDA environment
        env['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous execution
        env['CUDA_CACHE_DISABLE'] = '1'    # Disable CUDA cache
        
        return env 