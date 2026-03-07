"""Task-related helpers."""

from __future__ import annotations

from typing import Protocol


class _HasTaskFields(Protocol):
    task_id: str
    task_label: str | None


def format_task_ref(
    task: _HasTaskFields | str,
    *,
    include_label: bool = True,
    # short_id: bool = False,
    wrap: bool = True,
) -> str:
    """Return a human-readable identifier for a task.

    Args:
        task: Task-like object (must expose ``task_id`` / ``task_label``) or raw task ID string.
        include_label: Include ``task_label`` when available.
        short_id: Use abbreviated task id (first 8 chars).
        wrap: Surround segments with brackets (e.g. ``[label]``).
    """
    if isinstance(task, str):
        task_id = task
        task_label = None
    else:
        task_id = getattr(task, "task_id", "")
        task_label = getattr(task, "task_label", None)

    # task_id_display = task_id[:8] if short_id else task_id
    task_id_display = task_id
    if wrap:
        task_id_display = f"[{task_id_display}]"

    if include_label and task_label:
        label_part = f"[{task_label}]" if wrap else task_label
        separator = "::" if wrap else " :: "
        return f"{label_part}{separator}{task_id_display}"

    return task_id_display
