"""
GPU Information Module

Provides GPU hardware information and status queries.
"""

import subprocess
import os
import yaml
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Import configuration management from config module
from ..config.settings import load_config, get_gpu_id_mapping, reverse_gpu_id_mapping


class GPUType(Enum):
    """GPU type enumeration"""
    H100_PCIE_80G = "H100_PCIE_80G"
    L20 = "L20"
    RTX_6000_Ada = "RTX_6000_Ada"
    A100_PCIE_40G = "A100_PCIE_40G"
    UNKNOWN = "UNKNOWN"


@dataclass
class GPUSpec:
    """GPU specification data"""
    memory_gb: int
    compute_capability: str
    tensor_cores: bool
    max_power_w: int
    memory_bandwidth_gbps: int


@dataclass
class GPUInfo:
    """GPU information and status"""
    
    # Hardware identification
    device_id: int                    # CUDA device ID
    nvidia_smi_id: int               # nvidia-smi GPU ID
    name: str
    gpu_type: GPUType
    
    # Memory information
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    
    # Status information
    utilization_percent: int
    temperature_c: int
    power_draw_w: int
    
    # Specifications
    spec: GPUSpec
    
    @property
    def memory_usage_ratio(self) -> float:
        """Calculate memory usage ratio"""
        if self.memory_total_mb == 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb
    
    @property
    def is_available_for_functional(self) -> bool:
        """Check if GPU is available for functional tasks"""
        # Allow functional tasks if memory usage < 80% and temperature < 85C
        return (self.memory_usage_ratio < 0.8 and 
                self.temperature_c < 85 and
                self.utilization_percent < 95)
    
    @property
    def is_available_for_performance(self) -> bool:
        """Check if GPU is available for performance tasks"""
        # Require lower memory usage and utilization for performance tasks
        return (self.memory_usage_ratio < 0.1 and
                self.temperature_c < 80 and
                self.utilization_percent < 10)


def load_gpu_specs() -> Dict[GPUType, GPUSpec]:
    """从配置文件加载GPU规格"""
    from ..config.settings import get_gpu_specs
    gpu_specs_config = get_gpu_specs()
    
    gpu_specs = {}
    for gpu_type_str, spec_config in gpu_specs_config.items():
        try:
            gpu_type = GPUType(gpu_type_str)
            gpu_spec = GPUSpec(
                memory_gb=spec_config.get('memory_gb', 0),
                compute_capability=spec_config.get('compute_capability', '0.0'),
                tensor_cores=spec_config.get('tensor_cores', False),
                max_power_w=spec_config.get('max_power_w', 0),
                memory_bandwidth_gbps=spec_config.get('memory_bandwidth_gbps', 0)
            )
            gpu_specs[gpu_type] = gpu_spec
        except (ValueError, KeyError) as e:
            logger.warning(f"[GPUInfo] Failed to load GPU spec for {gpu_type_str}: {e}")
    
    # 如果配置文件加载失败，使用默认值
    if not gpu_specs:
        logger.info("[GPUInfo] Using default GPU specifications")
        gpu_specs = {
            GPUType.H100_PCIE_80G: GPUSpec(
                memory_gb=80,
                compute_capability="9.0",
                tensor_cores=True,
                max_power_w=350,  # H100 PCIe版本功耗
                memory_bandwidth_gbps=2000,
            ),
            GPUType.L20: GPUSpec(
                memory_gb=48,
                compute_capability="8.9",
                tensor_cores=True,
                max_power_w=275,
                memory_bandwidth_gbps=864,
            ),
            GPUType.RTX_6000_Ada: GPUSpec(
                memory_gb=48,
                compute_capability="8.9",
                tensor_cores=True,
                max_power_w=300,
                memory_bandwidth_gbps=960,
            ),
            GPUType.A100_PCIE_40G: GPUSpec(
                memory_gb=40,
                compute_capability="8.0",
                tensor_cores=True,
                max_power_w=250,  # A100 PCIe 40GB版本功耗
                memory_bandwidth_gbps=1555,  # A100 40GB版本显存带宽
            ),
            GPUType.UNKNOWN: GPUSpec(
                memory_gb=0,
                compute_capability="0.0",
                tensor_cores=False,
                max_power_w=0,
                memory_bandwidth_gbps=0,
            )
        }
    else:
        logger.info(f"[GPUInfo] Loaded {len(gpu_specs)} GPU specifications from config")
    
    return gpu_specs


# 动态加载GPU规格
GPU_SPECS = load_gpu_specs()


def _detect_gpu_type(gpu_name: str) -> GPUType:
    """Detect GPU type from name"""
    gpu_name_lower = gpu_name.lower()
    
    if 'h100' in gpu_name_lower:
        return GPUType.H100_PCIE_80G
    elif 'l20' in gpu_name_lower:
        return GPUType.L20
    elif 'rtx' in gpu_name_lower and '6000' in gpu_name_lower and 'ada' in gpu_name_lower:
        return GPUType.RTX_6000_Ada
    elif 'a100' in gpu_name_lower:
        return GPUType.A100_PCIE_40G
    else:
        logger.warning(f"[GPUInfo] Unknown GPU type: {gpu_name}")
        return GPUType.UNKNOWN


def query_gpu_info(available_gpu_ids: Optional[List[int]] = None) -> List[GPUInfo]:
    """Query GPU information from nvidia-smi"""
    if available_gpu_ids is None:
        available_gpu_ids = []  # Empty list means all GPUs are available
    
    try:
        # Query nvidia-smi for GPU information
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        gpu_infos = []
        cuda_device_id = 0
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue
            
            try:
                nvidia_smi_id = int(parts[0])
                
                # Only include available GPUs (if available_gpu_ids is specified)
                if available_gpu_ids and nvidia_smi_id not in available_gpu_ids:
                    continue
                
                name = parts[1]
                memory_total = int(parts[2])
                memory_used = int(parts[3])
                memory_free = int(parts[4])
                utilization = int(parts[5])
                temperature = int(parts[6])
                power_draw = int(float(parts[7]))  # Convert to int, handle float values
                
                gpu_type = _detect_gpu_type(name)
                spec = GPU_SPECS.get(gpu_type, GPU_SPECS[GPUType.UNKNOWN])
                
                gpu_info = GPUInfo(
                    device_id=cuda_device_id,
                    nvidia_smi_id=nvidia_smi_id,
                    name=name,
                    gpu_type=gpu_type,
                    memory_total_mb=memory_total,
                    memory_used_mb=memory_used,
                    memory_free_mb=memory_free,
                    utilization_percent=utilization,
                    temperature_c=temperature,
                    power_draw_w=power_draw,
                    spec=spec
                )
                
                gpu_infos.append(gpu_info)
                cuda_device_id += 1
                
            except (ValueError, IndexError) as e:
                logger.warning(f"[GPUInfo] Failed to parse GPU info line: {line}, error: {e}")
                continue
        
        if available_gpu_ids:
            logger.info(f"[GPUInfo] Found {len(gpu_infos)} GPUs from available list {available_gpu_ids}")
        else:
            logger.info(f"[GPUInfo] Found {len(gpu_infos)} available GPUs (all GPUs)")
        return gpu_infos
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[GPUInfo] nvidia-smi command failed: {e}")
        return []
    except Exception as e:
        logger.error(f"[GPUInfo] Failed to query GPU info: {e}")
        return []


def set_visible_gpus(gpu_ids: List[int]) -> None:
    """Set CUDA_VISIBLE_DEVICES environment variable"""
    if not gpu_ids:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        logger.info("[GPUInfo] Cleared CUDA_VISIBLE_DEVICES")
    else:
        cuda_ids = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_ids
        logger.info(f"[GPUInfo] Set CUDA_VISIBLE_DEVICES={cuda_ids}")


def get_available_gpus(available_gpu_ids: Optional[List[int]] = None) -> Dict[int, GPUInfo]:
    """Get available GPUs as dictionary mapped by CUDA device ID"""
    gpu_infos = query_gpu_info(available_gpu_ids)
    return {gpu.device_id: gpu for gpu in gpu_infos}


def get_all_gpus_info() -> Dict[int, GPUInfo]:
    """Get information for all system GPUs regardless of availability settings"""
    gpu_infos = query_gpu_info(available_gpu_ids=None)  # Get all GPUs
    return {gpu.device_id: gpu for gpu in gpu_infos} 