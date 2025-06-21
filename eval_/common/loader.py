"""
内核编译模块

负责CUDA和Triton内核的编译和加载
"""

import os
import importlib.util
from types import ModuleType
from pathlib import Path
from typing import Union, Callable

from torch.utils.cpp_extension import load

from .config import EvalConfig, DEFAULT_CONFIG


def _load_pyfile_module(pyfile_path: str, module_name: str = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name or Path(pyfile_path).stem, pyfile_path)
    module = importlib.util.module_from_spec(spec) # 名为module_name的module尚未加载
    spec.loader.exec_module(module) # 加载名为module_name的module, sys.modules[module_name] = module
    return module


def _load_pyfile_module_attr(pyfile_path: str, attr_name: str, module_name: str = None) -> Callable:
    module = _load_pyfile_module(pyfile_path, module_name)
    if hasattr(module, attr_name):
        return getattr(module, attr_name)
    else:
        return None


def load_cuda_extension_from_cufile(
    cuda_file: str,
    config: EvalConfig = DEFAULT_CONFIG
) -> Union[object, str]:
    if not os.path.exists(config.build_dir):
        os.makedirs(config.build_dir)
        
    try:
        extension = load(
            name=config.cuda_kernel_name,
            sources=[cuda_file],
            extra_cuda_cflags=config.extra_cuda_cflags,
            verbose=True,
            build_directory=config.build_dir
        )
        return extension
    except Exception as e:
        return str(e)

def load_cuda_extension_from_inline(
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