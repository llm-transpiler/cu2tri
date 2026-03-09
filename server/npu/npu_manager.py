"""NPU manager for monitoring and managing NPU resources."""
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from server.nvgpu.gpu_manager import GPUManager as _BaseGPUManager
from server.nvgpu.models import GPUMode, GPUStatus, Task
from server.nvgpu.config import config, TaskMode
from server.npu.logger import setup_logger
from server.common.task_refs import format_task_ref

if TYPE_CHECKING:
    from server.nvgpu.task_queue import TaskQueue
    from server.nvgpu.task_runner import TaskRunner

logger = setup_logger("npu_manager")

# Set by main.py
npu_config_loader = None


class NPUManager(_BaseGPUManager):
    """Ascend/CANN NPU manager built on existing scheduling logic."""

    def _initialize_nvml(self):
        """Initialize npu-smi based monitoring."""
        self.npu_smi_available = shutil.which("npu-smi") is not None
        if not self.npu_smi_available:
            logger.warning("npu-smi not available, NPU memory monitoring disabled")
            self.nvml_initialized = False
            self.nvml_device_count = 0
            return

        self.nvml_initialized = True
        self.nvml_device_count = self._detect_npu_count()
        logger.info("npu-smi detected, found %d NPUs", self.nvml_device_count)

    def _detect_npu_count(self) -> int:
        """Detect NPU count from npu-smi table."""
        output = self._run_npu_smi_info()
        if not output:
            return 0

        npu_ids: set[int] = set()
        # First table line format starts with: | 0     910B1 ...
        line_pattern = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|\s*\S+", re.MULTILINE)
        for match in line_pattern.finditer(output):
            npu_ids.add(int(match.group(1)))
        return len(npu_ids)

    def _run_npu_smi_info(self) -> str:
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except Exception as e:
            logger.debug("Failed to execute npu-smi info: %s", e)
            return ""

    def _parse_hbm_usage(self, output: str) -> dict[int, float]:
        """Parse HBM memory usage ratio by npu-smi id."""
        usage_by_id: dict[int, float] = {}
        current_npu_id: int | None = None

        head_line_re = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|")
        fraction_re = re.compile(r"(\d+)\s*/\s*(\d+)")

        bus_id_re = re.compile(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]")

        for line in output.splitlines():
            head_match = head_line_re.match(line)
            if head_match:
                current_npu_id = int(head_match.group(1))
                continue

            if current_npu_id is None or not line.lstrip().startswith("|"):
                continue

            # Only parse the second row of NPU status table (contains Bus-Id + HBM usage).
            if not bus_id_re.search(line):
                continue

            pairs = fraction_re.findall(line)
            if not pairs:
                continue

            # In second row, last fraction is typically HBM usage.
            used_str, total_str = pairs[-1]
            total = int(total_str)
            used = int(used_str)
            if total > 0:
                usage_by_id[current_npu_id] = used / total
            current_npu_id = None

        return usage_by_id

    def _update_gpu_memory(self, gpu_id: int):
        """Update NPU memory usage for logical device."""
        if not self.nvml_initialized:
            return

        output = self._run_npu_smi_info()
        if not output:
            return
        usage_by_smi_id = self._parse_hbm_usage(output)

        npu_smi_id = gpu_id
        if npu_config_loader:
            mapped_id = npu_config_loader.get_npu_smi_id(gpu_id)
            if mapped_id is not None:
                npu_smi_id = mapped_id

        usage = usage_by_smi_id.get(npu_smi_id)
        if usage is None:
            return

        with self.lock:
            if gpu_id in self.gpus:
                self.gpus[gpu_id].current_memory_usage = usage

    def register_npu(
        self,
        npu_id: int,
        mode: GPUMode = None,
        memory_threshold: float = None,
        max_concurrent_tasks: int = None,
    ) -> bool:
        return self.register_gpu(
            gpu_id=npu_id,
            mode=mode,
            memory_threshold=memory_threshold,
            max_concurrent_tasks=max_concurrent_tasks,
        )

    def list_npus(self):
        return self.list_gpus()

    def get_npu(self, npu_id: int):
        return self.get_gpu(npu_id)

    def set_npu_status(self, npu_id: int, status: GPUStatus) -> bool:
        return self.set_gpu_status(npu_id, status)

    def set_npu_mode(self, npu_id: int, mode: GPUMode, manual: bool = True) -> bool:
        return self.set_gpu_mode(npu_id, mode, manual=manual)

    def clear_npu_manual_mode(self, npu_id: int) -> bool:
        return self.clear_manual_mode(npu_id)

    def set_npu_memory_threshold(self, npu_id: int, threshold: float) -> bool:
        return self.set_gpu_memory_threshold(npu_id, threshold)

    def set_npu_max_concurrent_tasks(self, npu_id: int, max_tasks: int) -> bool:
        return self.set_gpu_max_concurrent_tasks(npu_id, max_tasks)

    def unregister_npu(self, npu_id: int) -> bool:
        return self.unregister_gpu(npu_id)

    def set_dependencies(self, task_queue: "TaskQueue", task_runner: "TaskRunner"):
        super().set_dependencies(task_queue, task_runner)

    def set_npu_mode_for_task(
        self,
        npu_id: int,
        task: Task,
        task_mode_override: GPUMode | None = None,
    ) -> bool:
        return self.set_gpu_mode_for_task(
            gpu_id=npu_id,
            task=task,
            task_mode_override=task_mode_override,
        )

    def restore_npu_mode_after_task(self, npu_id: int, task: Task) -> bool:
        return self.restore_gpu_mode_after_task(npu_id, task)

    def find_available_npu(self, preferred_npu: int | None = None, task_mode: TaskMode | None = None):
        return self.find_available_gpu(preferred_gpu=preferred_npu, task_mode=task_mode)

    def mark_task_running_on_npu(self, npu_id: int, task: Task) -> bool:
        return self.mark_task_running(gpu_id=npu_id, task=task)

    def mark_task_completed_on_npu(self, npu_id: int, task: Task) -> bool:
        return self.mark_task_completed(gpu_id=npu_id, task=task)

    def shutdown(self):
        """Shutdown manager."""
        self.stop_monitoring()
        logger.info("NPU manager shutdown completed")
