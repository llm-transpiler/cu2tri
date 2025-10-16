"""设备前缀计时示例。

演示如何使用 NVGPU/AMDGPU/BANG 封装区分不同宿主侧耗时。
"""

from __future__ import annotations

import time

from profiler.device_profilers import amdgpu_timer, bang_timer, nvgpu_timer


def simulate_work(duration: float) -> None:
    time.sleep(duration)


def main() -> None:
    nvgpu = nvgpu_timer()
    amdgpu = amdgpu_timer()
    bang = bang_timer()

    with nvgpu.time("launch_kernel"):
        simulate_work(0.01)

    with amdgpu.time("compile_module"):
        simulate_work(0.008)

    with bang.time("load_graph"):
        simulate_work(0.012)


if __name__ == "__main__":
    main()
