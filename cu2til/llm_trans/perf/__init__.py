from __future__ import annotations

"""Lightweight perf configuration helpers for cu2til.llm_trans.

This module keeps only the YAML/config related pieces. Execution logic for
NVGPU performance runs lives in ``cu2til.llm_trans.services.perf``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import yaml


_DEFAULT_DIRECTION = "cu2tri"
_DEFAULT_TESTSET = "xpiler"


def _config_path() -> Path:
    """Location of the perf configuration YAML (shared with services.perf)."""
    # perf/__init__.py → llm_trans/ → config/perf_runs.yaml
    return Path(__file__).resolve().parents[1] / "config" / "perf_runs.yaml"


def load_perf_config(path: Path | None = None) -> Mapping[str, object]:
    """Load performance configuration from YAML.

    The expected structure is::

        cu2tri:
          xpiler:
            model_tag:
              - 20250101_000000
              - 20250102_000000

    Returns an empty mapping if the file is missing or invalid.
    """
    cfg_path = path or _config_path()
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        if not isinstance(data, Mapping):
            return {}
        return data
    except Exception:
        # Be tolerant to partially written or invalid YAML.
        return {}


@dataclass(frozen=True)
class PerfRunSpec:
    """Single perf run specification resolved from config.yaml."""

    direction: str
    testset: str
    model_tag: str
    run_timestamp: str


def resolve_perf_specs_from_config(
    *,
    direction: str = _DEFAULT_DIRECTION,
    testset: str = _DEFAULT_TESTSET,
    config: Mapping[str, object] | None = None,
) -> List[PerfRunSpec]:
    """Resolve all configured perf run specs for the given direction/testset."""
    data = config or load_perf_config()
    dir_entry = data.get(direction)
    if not isinstance(dir_entry, Mapping):
        return []
    testset_entry = dir_entry.get(testset)
    if not isinstance(testset_entry, Mapping):
        return []

    specs: List[PerfRunSpec] = []
    for model_tag, timestamps in testset_entry.items():
        if not isinstance(model_tag, str):
            continue
        if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes)):
            for ts in timestamps:
                if isinstance(ts, str):
                    specs.append(
                        PerfRunSpec(
                            direction=direction,
                            testset=testset,
                            model_tag=model_tag,
                            run_timestamp=ts,
                        )
                    )
    return specs


__all__ = [
    "PerfRunSpec",
    "load_perf_config",
    "resolve_perf_specs_from_config",
]

