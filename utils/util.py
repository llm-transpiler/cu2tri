from typing import Any


def obj_to_dict(usage_obj: Any) -> dict | None:
    """Normalize completion usage (SDK object / pydantic / dict) to plain dict."""
    if usage_obj is None:
        return None
    try:
        if isinstance(usage_obj, dict):
            return usage_obj
        if hasattr(usage_obj, 'model_dump') and callable(getattr(usage_obj, 'model_dump')):
            return usage_obj.model_dump()
        if hasattr(usage_obj, '__dict__'):
            return {k: v for k, v in usage_obj.__dict__.items() if not k.startswith('_')}
        return dict(usage_obj)
    except Exception:
        return None
