"""设备相关的计时器封装。

这些封装基于 ``Timer``，在标签和输出中带上设备前缀，
便于区分 NVGPU、AMDGPU、BANG（寒武纪）等不同后端的宿主侧耗时。
"""

from __future__ import annotations

from typing import Callable

from .timer import HostTimer, TimerSample


class DeviceTimer(HostTimer):
    """带设备前缀的宿主侧计时器。"""

    def __init__(
        self,
        device_name: str,
        reporter: Callable[[TimerSample], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self.device_name = device_name
        super().__init__(reporter=reporter, enabled=enabled)

    def time(self, label: str):
        """为标签增加设备前缀再调用基类实现。"""
        prefixed = f"{self.device_name}:{label}"
        return super().time(prefixed)


def create_device_timer(
    device_name: str,
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> DeviceTimer:
    """通用工厂，适用于任意设备字符串。"""
    return DeviceTimer(device_name, reporter=reporter, enabled=enabled)


def nvgpu_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> DeviceTimer:
    """NVGPU（NVIDIA GPU）宿主侧计时器。"""
    return create_device_timer("NVGPU", reporter=reporter, enabled=enabled)


def amdgpu_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> DeviceTimer:
    """AMDGPU 宿主侧计时器。"""
    return create_device_timer("AMDGPU", reporter=reporter, enabled=enabled)


def bang_timer(
    reporter: Callable[[TimerSample], None] | None = None,
    enabled: bool = True,
) -> DeviceTimer:
    """BANG（寒武纪系列）宿主侧计时器。"""
    return create_device_timer("BANG", reporter=reporter, enabled=enabled)


__all__ = [
    "DeviceTimer",
    "create_device_timer",
    "nvgpu_timer",
    "amdgpu_timer",
    "bang_timer",
]
