from __future__ import annotations

from typing import Dict

from profiler.timer import Timer, TimerSample, create_timer
from utils.context import ManagedContext 


class TransTimer:
    """Cumulative, label-based timer for llm_trans phases.

    - Uses base Timer (create_timer) with perf_counter clock.
    - Supports start/stop for long phases and with-context for scoped timings.
    - Accumulates durations per label and exposes last measurement value.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._samples: list[TimerSample] = []
        self._durations_ms: Dict[str, float] = {}
        self._last_ms: Dict[str, float] = {}
        self._active: Dict[str, ManagedContext] = {}
        self._timer: Timer = create_timer(reporter=self._capture_sample, enabled=enabled)

    def _capture_sample(self, sample: TimerSample) -> None:
        # Update last and cumulative durations in milliseconds
        self._samples.append(sample)
        self._last_ms[sample.label] = sample.duration_ms
        self._durations_ms[sample.label] = self._durations_ms.get(sample.label, 0.0) + sample.duration_ms

    # Manual start/stop for phases that span across scopes
    def start(self, label: str) -> None:
        if label in self._active:
            return
        ctx = ManagedContext(self._timer.time(label))
        ctx.enter()
        self._active[label] = ctx

    def stop(self, label: str) -> float | None:
        ctx = self._active.pop(label, None)
        if not ctx:
            return None
        # Exiting context triggers reporter -> updates last & cumulative
        ctx.exit()
        return self._last_ms.get(label)

    # Scoped timing for single-block measurements
    def time(self, label: str):
        return self._timer.time(label)

    # Accessors
    def last_duration_ms(self, label: str) -> float | None:
        return self._last_ms.get(label)

    def total_duration_ms(self, label: str) -> float | None:
        return self._durations_ms.get(label)

    def as_dict(self) -> dict[str, float]:
        return dict(self._durations_ms)


