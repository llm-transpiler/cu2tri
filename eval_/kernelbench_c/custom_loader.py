from typing import Callable
from ..common.loader import _load_pyfile_module, _load_pyfile_module_attr
from types import ModuleType

def load_triton_kernel_from_pyfile(kernel_pyfile_path: str, attr_name: str = 'forward') -> Callable:
    return _load_pyfile_module_attr(kernel_pyfile_path, attr_name)


def load_torch_reference_from_pyfile(ref_pyfile_path: str, module_name: str = None) -> ModuleType:
    return _load_pyfile_module(ref_pyfile_path, module_name)
