from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping

import yaml


def _project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env)
    # Fallback: cu2til/llm_trans/services/cases.py → project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _load_case_config_yaml() -> dict:
    config_path = _project_root() / Path("cu2til/llm_trans/config/case_config.yaml")
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # Be tolerant: fall back to scanning when YAML invalid
        return {}


def _scan_cases(dir_rel_to_root: str, *, include_ops: list[str] | None = None, exclude_ops: list[str] | None = None) -> Dict[str, list[str]]:
    root = _project_root() / Path(dir_rel_to_root)
    if not root.exists():
        return {}
    include = set(include_ops or [])
    exclude = set(exclude_ops or [])

    grouped: Dict[str, list[str]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        op = path.name.split("_", 1)[0]
        if include and op not in include:
            continue
        if exclude and op in exclude:
            continue
        grouped.setdefault(op, []).append(path.name)

    # ensure deterministic order
    return {op: sorted(names) for op, names in sorted(grouped.items())}


def _resolve_from_yaml(testset: str) -> Dict[str, list[str]] | None:
    data = _load_case_config_yaml()
    testsets = data.get("testsets") if isinstance(data, dict) else None
    if not isinstance(testsets, dict):
        return None
    entry = testsets.get(testset)
    if not isinstance(entry, dict):
        return None

    mode = entry.get("mode", "scan")
    path = entry.get("path", "cu2til/cases/xpiler")
    include_ops = entry.get("include_ops") or []
    exclude_ops = entry.get("exclude_ops") or []

    if mode == "scan":
        return _scan_cases(path, include_ops=list(include_ops), exclude_ops=list(exclude_ops))
    if mode == "explicit":
        # expect explicit mapping under 'cases'
        cases = entry.get("cases")
        if isinstance(cases, dict):
            return {k: list(v) for k, v in cases.items() if isinstance(v, (list, tuple))}
    return None


def resolve_cases_for_testset(testset: str) -> Dict[str, list[str]]:
    # 1) Try YAML first
    resolved = _resolve_from_yaml(testset)
    if resolved is not None:
        return resolved

    # 2) Fallback: for xpiler* testsets, scan directory
    if testset.startswith("xpiler"):
        return _scan_cases("cu2til/cases/xpiler")

    raise ValueError(f"Unsupported testset: {testset}")


def first_case_each(case_map: Mapping[str, Iterable[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, values in case_map.items():
        iterator = iter(values)
        first = next(iterator, None)
        if first is not None:
            result[key] = first
    return result


def select_available_cases(
    all_cases: Mapping[str, Iterable[str]],
    *,
    first_only: bool,
    case_types: Iterable[str] | None,
    logger=None,
) -> Dict[str, Iterable[str]]:
    if first_only:
        base = first_case_each(all_cases)
    else:
        base = {k: list(v) for k, v in all_cases.items()}

    if not case_types:
        return base

    filtered: Dict[str, Iterable[str]] = {}
    for case_type in case_types:
        if case_type in base:
            filtered[case_type] = base[case_type]
        else:
            message = f"Case type '{case_type}' not found in available cases"
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
    return filtered


__all__ = [
    "resolve_cases_for_testset",
    "first_case_each",
    "select_available_cases",
]
