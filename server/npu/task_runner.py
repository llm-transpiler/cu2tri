"""Task runner for executing scripts on NPU."""
from pathlib import Path

import server.nvgpu.task_runner as _base_task_runner_mod
from server.nvgpu.task_runner import TaskRunner as _BaseTaskRunner

from server.npu.logger import setup_logger

logger = setup_logger("task_runner")

# Set by main.py
npu_config_loader = None


class TaskRunner(_BaseTaskRunner):
    """NPU-aware task runner using the existing subprocess pipeline."""

    def run_task(self, task, gpu_id: int) -> bool:
        # Redirect inherited log output location from nvgpu/ to npu/.
        _base_task_runner_mod.NVGPU_ROOT = Path(__file__).parent.resolve()
        _base_task_runner_mod.logger = logger

        # Keep existing Task model/scheduler wiring, but enforce
        # NPU-only visibility semantics for Ascend runtime.
        visible_id = gpu_id
        if npu_config_loader:
            mapped_id = npu_config_loader.get_visible_id(gpu_id)
            if mapped_id is not None:
                visible_id = mapped_id

        env = dict(task.env or {})
        # NPU-only default: runtime visible device selector.
        env.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(visible_id))
        task.env = env
        # Instruct base runner to avoid CUDA export and use Ascend runtime key only.
        task.disable_cuda_visible_devices = True
        task.visible_device_env_key = "ASCEND_RT_VISIBLE_DEVICES"

        return super().run_task(task, gpu_id)
