from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from server.common.timezone import ensure_timezone, now_timestamp

from ..core.runtime import RuntimeContext
from ..data.models import AttemptResult, BatchSummary, CaseResult
from ..io.jsonl import write_jsonl_log
from ..services.attempts import run_single_case_translation


async def run(context: RuntimeContext) -> Dict[str, object]:
    logger = context.logger
    settings = context.settings
    args = context.args
    available_cases = context.available_cases

    logger.info("🚀 Starting batch testing with async execution...")
    logger.info(f"📋 Available case types: {list(available_cases.keys())}")
    logger.info(f"📁 Work directory: {settings.work_dir}")
    logger.info(f"📊 Real-time statistics: {settings.jsonl_file}")

    if settings.jsonl_file is None:
        raise RuntimeError("JSONL log file path is not configured")

    _initialize_jsonl_metadata(context)

    logger.info("⚙️  Configuration:")
    logger.info(f"   - Model: {context.model.model_name}")
    logger.info(f"   - Base URL: {context.model.base_url}")
    logger.info(f"   - Temperature: {args.temperature}")
    logger.info(f"   - Testset: {args.testset}")
    logger.info(f"   - Max rounds: {settings.max_rounds}")
    logger.info(f"   - Console output: {settings.console_output}")
    logger.info(f"   - First only: {args.first_only}")
    logger.info(f"   - Skip performance: {not args.enable_perf}")
    logger.info(f"   - Retry wait time: {args.retry_wait}s")
    logger.info(f"   - Max retries: {args.max_retries}")
    logger.info(f"   - Multi-attempt: {args.max_attempts} ({args.attempt_policy})")
    logger.info(f"   - Millisecond format: {args.ms_format}")
    logger.info(
        f"   - Specific case types: {args.case_types if args.case_types else 'All'}"
    )
    logger.info(f"   - Use NPU server: {settings.use_nvgpu}")
    logger.info(f"   - Concurrency: {args.concurrency}")
    if settings.use_nvgpu:
        logger.info(f"   - NPU server: {args.nvgpu_server}")
        logger.info(f"   - NPU ID: {args.nvgpu_gpu or 'auto-assign'}")
        logger.info("   - NPU task type: functional")

    detailed_results: Dict[str, List[CaseResult]] = {}
    all_case_results: List[CaseResult] = []

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_case(case_type: str, case_name_single: str) -> Tuple[bool, int | None, List[AttemptResult]]:
        # Delay prominent start markers to actual attempt start; mark queueing at debug level if needed
        logger.debug(f"Queued {case_type}/{case_name_single}")
        return await run_single_case_translation(
            context,
            case_type,
            case_name_single,
            attempt_semaphore=semaphore if args.concurrency > 1 else None,
        )

    tasks: List[tuple[str, str, asyncio.Task]] = []
    for case_type, case_names in available_cases.items():
        if isinstance(case_names, (tuple, list)):
            names_iter: Iterable[str] = case_names
        else:
            names_iter = [case_names]

        detailed_results.setdefault(case_type, [])

        for case_name_single in names_iter:
            task = asyncio.create_task(run_case(case_type, case_name_single))
            tasks.append((case_type, case_name_single, task))

    results = await asyncio.gather(*(task for _, _, task in tasks), return_exceptions=True)

    for (case_type, case_name_single, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            case_result = CaseResult(
                case_type=case_type,
                case_name=case_name_single,
                success=False,
                rounds=None,
                attempt_details=[],
                successful_attempt=None,
                error=str(result),
            )
        else:
            success, rounds, attempt_details = result
            successful_attempt = next(
                (entry.attempt_number for entry in attempt_details if entry.success),
                None,
            )
            case_result = CaseResult(
                case_type=case_type,
                case_name=case_name_single,
                success=success,
                rounds=rounds,
                attempt_details=attempt_details,
                successful_attempt=successful_attempt,
                error=None,
            )
        detailed_results[case_type].append(case_result)
        all_case_results.append(case_result)

    _log_batch_summary(logger, detailed_results, settings)

    total_cases = sum(len(results) for results in detailed_results.values())
    total_success = sum(1 for result in all_case_results if result.success)
    rounds_distribution: Dict[int, int] = {}
    successful_cases = [result for result in all_case_results if result.success]

    for result in successful_cases:
        if result.rounds is None:
            continue
        rounds_distribution[result.rounds] = rounds_distribution.get(result.rounds, 0) + 1

    round_values = [result.rounds for result in successful_cases if result.rounds is not None]
    avg_rounds = sum(round_values) / len(round_values) if round_values else 0.0

    summary = BatchSummary(
        total_cases=total_cases,
        successful_cases=total_success,
        failed_cases=total_cases - total_success,
        success_rate=total_success / total_cases if total_cases else 0.0,
        case_types_total=len(detailed_results),
        case_types_fully_passed=sum(
            1 for _, results in detailed_results.items() if all(r.success for r in results)
        ),
        rounds_distribution=rounds_distribution,
        avg_rounds=avg_rounds,
        max_attempts=args.max_attempts,
        attempt_policy=args.attempt_policy,
        ms_format=args.ms_format,
        timestamp=settings.timestamp,
        completed_at=ensure_timezone(now_timestamp()).isoformat(),
    )
    await write_jsonl_log(context, summary.to_dict())

    return {
        "detailed_results": {
            case_type: [result.to_dict() for result in results]
            for case_type, results in detailed_results.items()
        },
        "all_case_results": [result.to_dict() for result in all_case_results],
        "summary": summary.to_dict(),
    }


def _initialize_jsonl_metadata(context: RuntimeContext) -> None:
    if context.settings.jsonl_file is None:
        return
    metadata = {
        "_type": "metadata",
        "model_name": context.model.model_name,
        "temperature": context.args.temperature,
        "testset": context.args.testset,
        "timestamp": context.settings.timestamp,
        "max_rounds": context.settings.max_rounds,
        "concurrency": context.args.concurrency,
        "use_nvgpu": context.settings.use_nvgpu,
        "nvgpu_server": context.args.nvgpu_server if context.settings.use_nvgpu else None,
        "enable_perf": context.args.enable_perf,
        "perf_warmup": getattr(context.args, "perf_warmup", None),
        "perf_iters": getattr(context.args, "perf_iters", None),
        "nvgpu_perf_gpu": getattr(context.args, "nvgpu_perf_gpu", None),
        "max_attempts": context.args.max_attempts,
        "attempt_policy": context.args.attempt_policy,
        "ms_format": context.args.ms_format,
        "created_at": ensure_timezone(now_timestamp()).isoformat(),
    }
    try:
        # Ensure parent directory exists
        context.settings.jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        with open(context.settings.jsonl_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        context.logger.debug(f"Initialized JSONL log file: {context.settings.jsonl_file}")
    except Exception as exc:  # pragma: no cover
        context.logger.warning(f"Failed to initialize JSONL log file: {exc}")


def _log_batch_summary(logger, detailed_results: Dict[str, List[CaseResult]], settings) -> None:
    logger.info("=" * 80)
    logger.info("📊 DETAILED BATCH TESTING SUMMARY")
    logger.info("=" * 80)

    for case_type, case_results in detailed_results.items():
        case_type_success = sum(1 for result in case_results if result.success)
        case_type_total = len(case_results)
        if case_type_success == case_type_total:
            case_type_status = "✅"
        elif case_type_success == 0:
            case_type_status = "❌"
        else:
            case_type_status = "⚠️ "
        logger.info(f"{case_type_status} {case_type:<15} ({case_type_success}/{case_type_total})")

        for result in case_results:
            attempt_count = len(result.attempt_details)
            if attempt_count == 0:
                logger.info(f"  ❌ (No attempts) {result.case_name}")
                continue

            if result.success:
                attempt_label = (
                    f"Attempt {result.successful_attempt}"
                    if result.successful_attempt is not None
                    else "Attempt ?"
                )
                round_label = (
                    f"Round {result.rounds}" if result.rounds is not None else "Round ?"
                )
                status = f"  ✅ ({attempt_label}, {round_label})"
            else:
                status = f"  ❌ (Attempts {attempt_count})"
            error_info = f" - {result.error}" if not result.success and result.error else ""
            logger.info(f"{status} {result.case_name}{error_info}")

            for attempt in result.attempt_details:
                attempt_status = "✅" if attempt.success else "❌"
                rounds_info = (
                    f"Round {attempt.rounds}" if attempt.rounds is not None else "No round"
                )
                logger.info(
                    f"    {attempt_status} Attempt {attempt.attempt_number}: {rounds_info}"
                )
        logger.info("")

    logger.info("=" * 80)
    logger.info(
        "🏆 OVERALL TOTAL: %s/%s individual cases succeeded",
        sum(1 for results in detailed_results.values() for r in results if r.success),
        sum(len(results) for results in detailed_results.values()),
    )
    logger.info(
        "📊 Case type success rate: %s/%s case types fully passed",
        sum(1 for _, results in detailed_results.items() if all(r.success for r in results)),
        len(detailed_results),
    )
    logger.info(f"📋 Log file: {settings.log_file}")
    logger.info(f"📊 Real-time statistics: {settings.jsonl_file}")
    logger.info("=" * 80)


__all__ = ["run"]
