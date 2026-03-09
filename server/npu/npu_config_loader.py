"""NPU configuration loader from YAML file."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from server.npu.logger import setup_logger

logger = setup_logger("npu_config")


@dataclass
class NPUConfig:
    """NPU configuration from YAML."""

    logical_id: int
    npu_smi_id: int
    visible_id: int
    name: str
    uuid: str | None = None
    memory_gb: float | None = None
    enabled: bool = True
    default_mode: str = "shared"
    memory_threshold: float = 0.75
    max_concurrent_tasks: int = 3


class NPUConfigLoader:
    """Load NPU configuration from YAML file."""

    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.npu_configs: dict[int, NPUConfig] = {}
        self.server_config: dict = {}

    def load(self) -> bool:
        if not self.config_file.exists():
            logger.warning("NPU config file not found: %s", self.config_file)
            return False

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.error("Empty NPU configuration file")
                return False

            self.npu_configs.clear()
            npus_data = config_data.get("npus")
            if npus_data is None:
                # Backward compatibility with old key.
                npus_data = config_data.get("gpus", [])

            for npu_data in npus_data:
                logical_id = npu_data["logical_id"]
                # Compatibility: accept old GPU keys as fallbacks.
                npu_smi_id = npu_data.get("npu_smi_id", npu_data.get("nvidia_smi_id", logical_id))
                visible_id = npu_data.get("visible_id", npu_data.get("cuda_visible_id", logical_id))
                npu_config = NPUConfig(
                    logical_id=logical_id,
                    npu_smi_id=npu_smi_id,
                    visible_id=visible_id,
                    name=npu_data.get("name", "Unknown"),
                    uuid=npu_data.get("uuid"),
                    memory_gb=npu_data.get("memory_gb"),
                    enabled=npu_data.get("enabled", True),
                    default_mode=npu_data.get("default_mode", "shared"),
                    memory_threshold=npu_data.get("memory_threshold", 0.75),
                    max_concurrent_tasks=npu_data.get("max_concurrent_tasks", 3),
                )
                self.npu_configs[logical_id] = npu_config

            self.server_config = config_data.get("server", {})

            logger.info("Loaded NPU configuration: %d NPUs", len(self.npu_configs))
            return True
        except Exception as e:
            logger.error("Failed to load NPU configuration: %s", e, exc_info=True)
            return False

    def get_npu_config(self, logical_id: int) -> NPUConfig | None:
        return self.npu_configs.get(logical_id)

    def get_enabled_npus(self) -> list[NPUConfig]:
        return [npu for npu in self.npu_configs.values() if npu.enabled]

    def get_visible_id(self, logical_id: int) -> int | None:
        npu_config = self.npu_configs.get(logical_id)
        return npu_config.visible_id if npu_config else None

    def get_npu_smi_id(self, logical_id: int) -> int | None:
        npu_config = self.npu_configs.get(logical_id)
        return npu_config.npu_smi_id if npu_config else None

    # Compatibility with inherited nvgpu task runner/manager hooks.
    def get_cuda_visible_id(self, logical_id: int) -> int | None:
        return self.get_visible_id(logical_id)

    def get_nvidia_smi_id(self, logical_id: int) -> int | None:
        return self.get_npu_smi_id(logical_id)

    def should_auto_register(self) -> bool:
        return self.server_config.get("auto_register_npus", self.server_config.get("auto_register_gpus", True))
