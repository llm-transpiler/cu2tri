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
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            set_env()
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            available_devices = []
            
            cnt = 0
            i = 0
            compute_capability = set()
            while len(available_devices) < device_count and cnt < 10:
                try:
                    props = torch.cuda.get_device_properties(i)
                    # 尝试设置设备，如果失败则跳过
                    torch.cuda.set_device(i)
                    
                    compute_capability.add(int(f"{props.major}{props.minor}"))
                    available_devices.append(i)
                    i = i + 1
                    
                except RuntimeError as e:
                    # 设备不可用，跳过
                    pass
                cnt = cnt + 1
            
            if compute_capability:
                # 返回最高的计算能力
                return max(compute_capability)
            else:
                return 90  # 默认值
        else:
            return 90  # 默认值
    except Exception:
        return 90  # 默认值

@dataclass
class EvalConfig:
    """评估配置类"""
    
    # 编译参数
    extra_cuda_cflags: List[str] = None
    build_dir: str = './build'
    cuda_kernel_name: str = "cuda_kernel"
    # 自动检测CUDA计算能力
    cuda_arch_number: int = _get_cuda_compute_capability()
    
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
        flag_template = '-gencode=arch=compute_{arch_number},code=sm_{arch_number}'
        if self.extra_cuda_cflags is None:
            self.extra_cuda_cflags = [
                '-O3', 
                '--use_fast_math', 
                flag_template.format(arch_number=self.cuda_arch_number),
            ]


# 默认配置实例
DEFAULT_CONFIG = EvalConfig()