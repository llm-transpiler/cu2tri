"""GPU configuration loader from YAML file."""
import yaml
from pathlib import Path
from dataclasses import dataclass

from logger import setup_logger

logger = setup_logger("gpu_config")


@dataclass
class GPUConfig:
    """GPU configuration from YAML."""
    logical_id: int
    nvidia_smi_id: int
    cuda_visible_id: int
    name: str
    uuid: str | None = None
    memory_gb: float | None = None
    enabled: bool = True
    default_mode: str = "shared"
    memory_threshold: float = 0.75
    max_concurrent_tasks: int = 3


class GPUConfigLoader:
    """Load GPU configuration from YAML file."""
    
    def __init__(self, config_file: str = "configs/gpu_resource.yml"):
        """Initialize loader.
        
        Args:
            config_file: Path to GPU configuration YAML file
        """
        self.config_file = Path(config_file)
        self.gpu_configs: dict[int, GPUConfig] = {}
        self.server_config: dict = {}
    
    def load(self) -> bool:
        """Load GPU configuration from YAML file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.config_file.exists():
            logger.warning(f"GPU config file not found: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.error("Empty GPU configuration file")
                return False
            
            # Load GPU configurations
            gpus_data = config_data.get('gpus', [])
            for gpu_data in gpus_data:
                gpu_config = GPUConfig(
                    logical_id=gpu_data['logical_id'],
                    nvidia_smi_id=gpu_data.get('nvidia_smi_id', gpu_data['logical_id']),
                    cuda_visible_id=gpu_data.get('cuda_visible_id', gpu_data['logical_id']),
                    name=gpu_data.get('name', 'Unknown'),
                    uuid=gpu_data.get('uuid'),
                    memory_gb=gpu_data.get('memory_gb'),
                    enabled=gpu_data.get('enabled', True),
                    default_mode=gpu_data.get('default_mode', 'shared'),
                    memory_threshold=gpu_data.get('memory_threshold', 0.75),
                    max_concurrent_tasks=gpu_data.get('max_concurrent_tasks', 3)
                )
                self.gpu_configs[gpu_config.logical_id] = gpu_config
            
            # Load server configuration
            self.server_config = config_data.get('server', {})
            
            logger.info(f"Loaded GPU configuration: {len(self.gpu_configs)} GPUs")
            for gpu_id, gpu_config in self.gpu_configs.items():
                logger.info(f"  GPU {gpu_id}: {gpu_config.name} "
                           f"(nvidia-smi={gpu_config.nvidia_smi_id}, "
                           f"cuda_vis={gpu_config.cuda_visible_id}, "
                           f"enabled={gpu_config.enabled})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GPU configuration: {e}")
            return False
    
    def get_gpu_config(self, logical_id: int) -> GPUConfig | None:
        """Get GPU configuration by logical ID.
        
        Args:
            logical_id: Logical GPU ID
            
        Returns:
            GPUConfig or None if not found
        """
        return self.gpu_configs.get(logical_id)
    
    def get_enabled_gpus(self) -> list[GPUConfig]:
        """Get list of enabled GPUs.
        
        Returns:
            List of enabled GPUConfig objects
        """
        return [gpu for gpu in self.gpu_configs.values() if gpu.enabled]
    
    def get_cuda_visible_id(self, logical_id: int) -> int | None:
        """Get CUDA_VISIBLE_DEVICES ID for a logical GPU ID.
        
        Args:
            logical_id: Logical GPU ID
            
        Returns:
            CUDA visible ID or None if not found
        """
        gpu_config = self.gpu_configs.get(logical_id)
        return gpu_config.cuda_visible_id if gpu_config else None
    
    def get_nvidia_smi_id(self, logical_id: int) -> int | None:
        """Get nvidia-smi ID for a logical GPU ID.
        
        Args:
            logical_id: Logical GPU ID
            
        Returns:
            nvidia-smi ID or None if not found
        """
        gpu_config = self.gpu_configs.get(logical_id)
        return gpu_config.nvidia_smi_id if gpu_config else None
    
    def should_auto_register(self) -> bool:
        """Check if GPUs should be auto-registered on startup.
        
        Returns:
            True if auto-registration is enabled
        """
        return self.server_config.get('auto_register_gpus', True)

