from __future__ import annotations

import asyncio
from typing import Sequence

from .clients import NVGPU_AVAILABLE, ModelClients, create_model_clients
from .config import Settings, build_settings
from .core.runtime import RuntimeContext
from .io.logging import configure_logger
from .services import cases
from .services.runner import run


def prepare_context(argv: Sequence[str] | None = None) -> RuntimeContext:
    settings: Settings = build_settings(list(argv) if argv is not None else None)
    model_clients: ModelClients = create_model_clients(settings)
    settings.configure_output_paths(model_clients.model_name)

    logger = configure_logger(settings)

    available_cases = cases.select_available_cases(
        settings.all_cases,
        first_only=getattr(settings.args, "first_only", False),
        case_types=getattr(settings.args, "case_types", None),
        skip_case_types=getattr(settings.args, "skip_case_types", None),
        logger=logger,
    )
    
    if settings.use_nvgpu and not NVGPU_AVAILABLE:
        logger.warning("NPU/NVGPU client not available, falling back to local execution")
        settings.use_nvgpu = False

    context = RuntimeContext(
        settings=settings,
        model=model_clients,
        logger=logger,
        available_cases=available_cases,
        nvgpu_available=settings.use_nvgpu,
    )

    return context


def main(argv: Sequence[str] | None = None) -> None:
    context = prepare_context(argv)
    if not context.available_cases:
        context.logger.error("No cases available for testing. Check your case types filter.")
        raise SystemExit(1)
    asyncio.run(run(context))


if __name__ == "__main__":  # pragma: no cover
    main()
