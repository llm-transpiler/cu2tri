import os
import importlib
import traceback
from pathlib import Path
from typing import Tuple, Optional
import logging
from utils.config_utils import KernelInfo, TestResult
from utils.io_utils import capture_output
from torch.utils.cpp_extension import load_inline

class KernelCompiler:
    """Safe kernel compilation with detailed error collection"""
    
    @staticmethod
    def safe_compile_kernel(kernel_info: KernelInfo, verbose: bool = True, logger: logging.Logger = None) -> Tuple[TestResult, Optional[callable]]:
        assert logger is not None
        """Safely compile kernel with comprehensive error handling"""
        result = TestResult(kernel_name=kernel_info.name)
        
        logger.info(f"=== Compile {kernel_info.name} Kernel ===")
        
        try:
            with capture_output() as (stdout_capture, stderr_capture):
                if kernel_info.kernel_type.startswith('cuda'):
                    kernel_func = KernelCompiler._compile_cuda_kernel_inline(kernel_info, verbose)
                else:
                    kernel_func = KernelCompiler._compile_other_kernel(kernel_info)
            
            result.compile_stdout = stdout_capture.getvalue()
            result.compile_stderr = stderr_capture.getvalue()
            result.compile_success = True
            result.compile_message = f"{kernel_info.name} Compilation Success"
            
            logger.info(f"✓ {kernel_info.name} Compilation Success")
            if verbose and result.compile_stdout.strip():
                logger.info(f"Compilation Output: {result.compile_stdout.strip()}")
            
            return result, kernel_func
            
        except ImportError as e:
            result.compile_error = str(e)
            result.compile_message = f"{kernel_info.name} Module Unavailable: {str(e)}"
            logger.info(f"✗ {result.compile_message}")
            return result, None
            
        except Exception as e:
            result.compile_error = str(e)
            result.compile_message = f"{kernel_info.name} Compilation Failed: {str(e)}"
            logger.info(f"✗ {result.compile_message}")
            logger.info(f"Error Details: {traceback.format_exc()}")
            return result, None
    
    @staticmethod
    def _compile_cuda_kernel_inline(kernel_info: KernelInfo, verbose: bool) -> callable:
        """Compile CUDA kernel"""
        
        
        # Load module
        spec = importlib.util.spec_from_file_location(
            Path(kernel_info.module_path).stem,
            kernel_info.module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Prepare build directory
        build_dir = f'./build/{kernel_info.name}'
        os.makedirs(build_dir, exist_ok=True)
        
        # Get source code
        cpp_source = getattr(module, 'cpp_declaration_source', '')
        cuda_source = getattr(module, 'cuda_kernel_source', '') + \
                     getattr(module, 'cuda_wrapper_source', '')
        
        # Compilation options
        extra_cflags = ["-O3"]
        extra_ldflags = ["-lcudnn", "-lcublas"]
        
        # Compile
        kernel_impl = load_inline(
            name=f"{kernel_info.callable_name}_{kernel_info.name}",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[kernel_info.callable_name],
            verbose=verbose,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=build_dir
        )
        
        return getattr(kernel_impl, kernel_info.callable_name)
    
    @staticmethod
    def _compile_other_kernel(kernel_info: KernelInfo) -> callable:
        """Compile other kernels (Triton, etc.)"""
        spec = importlib.util.spec_from_file_location(
            Path(kernel_info.module_path).stem,
            kernel_info.module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return getattr(module, kernel_info.callable_name)

