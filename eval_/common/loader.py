"""
内核编译模块

负责CUDA和Triton内核的编译和加载
"""

import os
from types import ModuleType
from pathlib import Path
from typing import Union, Callable

from torch.utils.cpp_extension import load

from .config import EvalConfig, DEFAULT_CONFIG


def _load_pyfile_module(pyfile_path: str, module_name: str = None) -> ModuleType:
    import importlib.util
    import sys
    if module_name is None:
        module_name = Path(pyfile_path).stem
    
    spec = importlib.util.spec_from_file_location(module_name, pyfile_path)
    module = importlib.util.module_from_spec(spec)
    
    # 保存之前的模块（如果存在）以备回滚
    old_module = sys.modules.get(module_name)
    sys.modules[module_name] = module # 多轮测试如果用了同一个名字可以覆盖加载
    
    try:
        spec.loader.exec_module(module) # 加载名为module_name的module
    except Exception:
        # 执行失败时回滚
        if old_module is not None:
            sys.modules[module_name] = old_module
        else:
            sys.modules.pop(module_name, None)
        raise
    
    return module


def _load_pyfile_module_attr(pyfile_path: str, attr_name: str, module_name: str = None) -> Callable:
    module = _load_pyfile_module(pyfile_path, module_name)
    if hasattr(module, attr_name):
        return getattr(module, attr_name)
    else:
        return None


def load_cuda_extension_from_cufile(
    cuda_file: str | list[str],
    config: EvalConfig = DEFAULT_CONFIG()
) -> Union[object, str]:
    if not os.path.exists(config.build_dir):
        os.makedirs(config.build_dir)
        
    try:
        extension = load(
            name=getattr(config, 'cuda_kernel_name', 'cuda_kernel'),
            sources=cuda_file if isinstance(cuda_file, list) or isinstance(cuda_file, tuple) else [cuda_file],
            extra_cuda_cflags=getattr(config, 'extra_cuda_cflags', []),
            extra_ldflags=getattr(config, 'extra_ldflags', []),
            verbose=getattr(config, 'verbose', True),
            build_directory=getattr(config, 'build_dir', './build')
        )
        return extension
    except Exception as e:
        return str(e)

def load_cuda_extension_inline(
    name: str,
    cpp_sources: str | list[str],
    cuda_sources: str | list[str],
    functions: list[str] = [],
    extra_cuda_cflags: list[str] = [],
    build_directory: str = "./build",
    verbose: bool = False
) -> Union[object, str]:
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name=name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_directory,
        verbose=verbose
    )