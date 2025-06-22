"""
GPU Information Management Module

Defines GPU types, specifications, and information retrieval functions.
"""

import os
import subprocess
import yaml
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "config.yaml"


def load_config() -> Dict:
    """加载配置文件"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config file {CONFIG_FILE}: {e}")
        return {}


def get_gpu_id_mapping() -> Dict[int, int]:
    """获取GPU ID映射关系 (CUDA device ID -> nvidia-smi device ID)"""
    config = load_config()
    return config.get('gpu_id_mapping', {})


def reverse_gpu_id_mapping() -> Dict[int, int]:
    """获取反向GPU ID映射关系 (nvidia-smi device ID -> CUDA device ID)"""
    mapping = get_gpu_id_mapping()
    return {v: k for k, v in mapping.items()}


class GPUType(Enum):
    """GPU type enumeration"""
    H100 = "H100"
    H100_80GB = "H100_80GB"
    L20 = "L20"
    RTX_6000_ADA = "RTX_6000_Ada"
    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"
    A100 = "A100"
    A6000 = "RTX_6000_Ada"
    UNKNOWN = "Unknown"


@dataclass
class GPUSpec:
    """GPU specification information"""
    memory_gb: int
    compute_capability: str
    tensor_cores: bool
    max_power_w: int
    memory_bandwidth_gbps: int
    fp32_tflops: float


def load_gpu_specs() -> Dict[GPUType, GPUSpec]:
    """从配置文件加载GPU规格"""
    config = load_config()
    gpu_specs_config = config.get('gpu_specs', {})
    
    gpu_specs = {}
    for gpu_type_str, spec_config in gpu_specs_config.items():
        try:
            gpu_type = GPUType(gpu_type_str)
            gpu_spec = GPUSpec(
                memory_gb=spec_config.get('memory_gb', 0),
                compute_capability=spec_config.get('compute_capability', '0.0'),
                tensor_cores=spec_config.get('tensor_cores', False),
                max_power_w=spec_config.get('max_power_w', 0),
                memory_bandwidth_gbps=spec_config.get('memory_bandwidth_gbps', 0),
                fp32_tflops=spec_config.get('fp32_tflops', 0.0)
            )
            gpu_specs[gpu_type] = gpu_spec
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to load GPU spec for {gpu_type_str}: {e}")
    
    # 如果配置文件加载失败，使用默认值
    if not gpu_specs:
        gpu_specs = {
            GPUType.H100: GPUSpec(
                memory_gb=80,
                compute_capability="9.0",
                tensor_cores=True,
                max_power_w=350,
                memory_bandwidth_gbps=2000,
                fp32_tflops=51.0
            ),
            GPUType.L20: GPUSpec(
                memory_gb=48,
                compute_capability="8.6",
                tensor_cores=True,
                max_power_w=350,
                memory_bandwidth_gbps=864,
                fp32_tflops=59.7
            ),
            GPUType.RTX_6000_ADA: GPUSpec(
                memory_gb=48,
                compute_capability="8.9",
                tensor_cores=True,
                max_power_w=300,
                memory_bandwidth_gbps=960,
                fp32_tflops=91.1
            ),
            GPUType.A100_40GB: GPUSpec(
                memory_gb=40,
                compute_capability="8.0",
                tensor_cores=True,
                max_power_w=300,
                memory_bandwidth_gbps=960,
                fp32_tflops=91.1
            ),
            GPUType.A100_80GB: GPUSpec(
                memory_gb=80,
                compute_capability="8.0",
                tensor_cores=True,
                max_power_w=300,
                memory_bandwidth_gbps=960,
                fp32_tflops=91.1
            )
        }
    
    return gpu_specs


# 动态加载GPU规格
GPU_SPECS = load_gpu_specs()


@dataclass
class GPUInfo:
    """GPU device information"""
    device_id: int  # CUDA device ID (用于PyTorch)
    nvidia_smi_id: int  # nvidia-smi device ID (用于nvidia-smi查询)
    gpu_type: GPUType
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int
    temperature_c: int
    power_draw_w: int
    
    @property
    def memory_usage_ratio(self) -> float:
        """Calculate memory usage ratio (0.0 to 1.0)"""
        return self.memory_used_mb / self.memory_total_mb if self.memory_total_mb > 0 else 0.0
    
    @property
    def is_available_for_functional_task(self) -> bool:
        """Check if GPU is available for functional tasks (memory usage < 2/3)"""
        config = load_config()
        limit_ratio = config.get('task_queue', {}).get('functional_memory_limit_ratio', 0.67)
        return self.memory_usage_ratio < limit_ratio
    
    @property
    def is_available_for_performance_task(self) -> bool:
        """Check if GPU is available for performance tasks (exclusive use)"""
        config = load_config()
        limit_ratio = config.get('task_queue', {}).get('performance_memory_limit_ratio', 0.05)
        return self.memory_usage_ratio < limit_ratio
    
    @property
    def spec(self) -> GPUSpec:
        """Get GPU specification"""
        return GPU_SPECS.get(self.gpu_type, GPUSpec(0, "0.0", False, 0, 0, 0.0))


def detect_gpu_type(gpu_name: str) -> GPUType:
    """Detect GPU type from name string"""
    name_upper = gpu_name.upper()
    
    if "H100" in name_upper:
        return GPUType.H100
    elif "L20" in name_upper:
        return GPUType.L20
    elif "RTX 6000 ADA" in name_upper or "RTX6000ADA" in name_upper or ("6000" in name_upper and "Ada" in name_upper):
        return GPUType.RTX_6000_ADA
    elif "A100" in name_upper and "40GB" in name_upper:
        return GPUType.A100_40GB
    elif "A100" in name_upper and "80GB" in name_upper:
        return GPUType.A100_80GB
    elif "A100" in name_upper:
        return GPUType.A100
    else:
        return GPUType.UNKNOWN


def get_visible_gpu_ids() -> List[int]:
    """Get list of visible GPU IDs based on CUDA_VISIBLE_DEVICES (返回CUDA device IDs)"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if not cuda_visible:
        # 如果没有设置，使用配置文件中的映射关系推断所有可用GPU
        gpu_mapping = get_gpu_id_mapping()
        if gpu_mapping:
            return list(gpu_mapping.keys())
        else:
            # 如果配置文件也没有，尝试检测所有GPU
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, check=True)
                return list(range(len(result.stdout.strip().split('\n'))))
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Failed to detect GPUs with nvidia-smi")
                return []
    
    # Parse CUDA_VISIBLE_DEVICES
    try:
        return [int(gpu_id.strip()) for gpu_id in cuda_visible.split(',') if gpu_id.strip()]
    except ValueError:
        logger.error(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible}")
        return []


def query_gpu_info() -> List[GPUInfo]:
    """Query current GPU information using nvidia-smi with ID mapping support"""
    visible_cuda_ids = get_visible_gpu_ids()
    if not visible_cuda_ids:
        return []
    
    # 获取GPU ID映射关系
    # cuda_to_nvidia_mapping = get_gpu_id_mapping()
    nvidia_to_cuda_mapping = reverse_gpu_id_mapping()
    
    gpu_infos = []
    
    try:
        # Query GPU information from nvidia-smi
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 7:
                continue
            
            try:
                nvidia_smi_id = int(parts[0])  # nvidia-smi device ID
                
                # 查找对应的CUDA device ID
                cuda_device_id = nvidia_to_cuda_mapping.get(nvidia_smi_id)
                
                # 如果没有映射关系，假设ID相同
                if cuda_device_id is None:
                    cuda_device_id = nvidia_smi_id
                
                # 只包含可见的CUDA设备
                if cuda_device_id not in visible_cuda_ids:
                    continue
                
                name = parts[1]
                memory_total = int(parts[2])
                memory_used = int(parts[3])
                utilization = int(parts[4]) if parts[4] != 'N/A' else 0
                temperature = int(parts[5]) if parts[5] != 'N/A' else 0
                power_draw = float(parts[6]) if parts[6] != 'N/A' else 0.0
                
                gpu_type = detect_gpu_type(name)
                
                gpu_info = GPUInfo(
                    device_id=cuda_device_id,  # CUDA device ID
                    nvidia_smi_id=nvidia_smi_id,  # nvidia-smi device ID
                    gpu_type=gpu_type,
                    name=name,
                    memory_total_mb=memory_total,
                    memory_used_mb=memory_used,
                    utilization_percent=utilization,
                    temperature_c=temperature,
                    power_draw_w=int(power_draw)
                )
                
                gpu_infos.append(gpu_info)
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse GPU info line: {line}, error: {e}")
                continue
    
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to query GPU information: {e}")
        return []
    
    return gpu_infos


def set_visible_gpus(gpu_ids: List[int]) -> None:
    """Set CUDA_VISIBLE_DEVICES environment variable (使用CUDA device IDs)"""
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    logger.info(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}") 