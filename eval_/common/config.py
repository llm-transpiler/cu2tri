"""
评估配置模块

管理环境设置、测试参数等配置项
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from utils.set_env import set_env


def _get_cuda_compute_capability() -> int:
    """获取当前CUDA设备的计算能力"""
    try:
        with torch.no_grad():
            import torch
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                set_env()
            # 延迟导入torch，避免在模块级别导入时就占用GPU
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                compute_capability = set()
                
                # 只查询设备属性，不设置当前设备，避免创建CUDA context
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        compute_capability.add(int(f"{props.major}{props.minor}"))
                    except RuntimeError as e:
                        # 设备不可用，跳过
                        continue
                
                if compute_capability:
                    # 返回最高的计算能力
                    return sorted(list(compute_capability))
                else:
                    return [80,86,89,90]  # 默认值
            else:
                return [80,86,89,90]  # 默认值
    except Exception:
        return [80,86,89,90]  # 默认值

@dataclass
class EvalConfig:
    """评估配置类"""
    
    # 编译参数
    extra_cuda_cflags: List[str] = None
    build_dir: str = './build'
    cuda_kernel_name: str = "cuda_kernel"
    # 延迟初始化CUDA计算能力，避免在模块导入时就检测GPU
    cuda_arch_number: list[int] = None
    
    # 性能测试参数
    warmup_runs: int = 1000
    test_runs: int = 10000
    
    # 超时设置
    subproc_timeout: int = 300
    
    # 容忍度设置
    atol: float = 1e-2
    rtol: float = 1e-2
    
    # 日志设置
    log_level: str = "DEBUG"
    
    def __post_init__(self):
        # 延迟检测CUDA计算能力
        if self.cuda_arch_number is None:
            self.cuda_arch_number = _get_cuda_compute_capability()
        if self.extra_cuda_cflags is None:
            self.extra_cuda_cflags = [
                '-O3', 
                '--use_fast_math', 
            ]
        for arch_number in self.cuda_arch_number:
            flag_template = '-gencode=arch=compute_{arch_number},code=sm_{arch_number}'
            self.extra_cuda_cflags.append(flag_template.format(arch_number=arch_number))

DEFAULT_CONFIG = EvalConfig()