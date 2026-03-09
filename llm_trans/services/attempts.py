from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import json

from ..utils.trans_timer import TransTimer
from server.common.timezone import ensure_timezone, now_timestamp

from llm_trans.prompts.cuda2triton import simple_initial_prompt as cuda2triton_prompt
from llm_trans.prompts.cuda2ascendc import simple_initial_prompt as cuda2ascendc_prompt

from ..clients import async_openai_llm_call
from ..core.runtime import RuntimeContext
from ..data.models import AttemptResult, AttemptTimingStats, RetryRecord, RoundRecord
from ..io.jsonl import write_jsonl_log
from ..services.conversation import (
    get_last_code_block,
    save_conversation_history,
    save_llm_conversation,
)
from ..services.history import AttemptHistoryManager
from ..services.retry import build_retry_context, is_retryable_error, log_retry_event
from ..utils.formatting import format_ms
from ..services.testing import run_testing_loop


async def run_single_case_attempt(
    context: RuntimeContext,
    case_type: str,
    case_name: str,
    attempt_number: int,
    attempt_work_dir: Path,
) -> tuple[bool, int | None, AttemptTimingStats]:
    logger = context.logger
    settings = context.settings
    args = context.args
    model_name = context.model.model_name
    async_client = context.model.async_client
    get_api_param = context.model.get_api_param

    case_start_wall = now_timestamp()
    trans_timer = TransTimer()
    trans_timer.start("case_total")

    logger.info("=" * 60)
    logger.info(f"🎯 Testing case type: {case_type}")
    logger.info(f"📁 Case name: {case_name}")
    if args.max_attempts > 1:
        logger.info(f"🧪 Attempt {attempt_number}/{args.max_attempts}")
    logger.info("=" * 60)

    testcase_src_dir = settings.testset_root_dir / case_name
    if attempt_work_dir.exists():
        if not settings.resume_conversation:
            shutil.rmtree(attempt_work_dir)
            attempt_work_dir.mkdir(parents=True, exist_ok=True)
    else:
        attempt_work_dir.mkdir(parents=True, exist_ok=True)

    timing_stats = AttemptTimingStats(
        case_type=case_type,
        case_name=case_name,
        attempt_number=attempt_number,
        start_time=ensure_timezone(case_start_wall).isoformat(),
        temperature=getattr(args, 'temperature', None),
    )

    # Determine which files to copy based on translation direction
    direction = settings.direction
    
    files_to_copy = [
        settings.dir_torch / "ref.py",
        "get_data.py",
    ]
    
    # Add files based on direction
    if direction == "cu2tri":
        # CUDA to Triton
        files_to_copy.extend([
            settings.dir_cuda / "kernel.cu",
            f"check_cuda{settings.check_suffix}.py",
            f"check_triton{settings.check_suffix}.py",
        ])
    elif direction == "cu2asc":
        # CUDA to Ascend C
        files_to_copy.extend([
            settings.dir_cuda / "kernel.cu",
        ])
    elif direction == "tri2cute":
        # Triton to CUTE
        files_to_copy.extend([
            settings.dir_triton / "kernel.py",
            settings.dir_cute / "kernel.cu",
            settings.dir_cute / "kernel_template.cu",
            "check_cute.py",
        ])

    for file_path in files_to_copy:
        src_file = testcase_src_dir / file_path
        dst_file = attempt_work_dir / file_path

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            logger.debug(f"Copy {file_path} to {attempt_work_dir}")
        else:
            logger.warning(f"Source file {src_file} does not exist")

    # Always provide the latest tool scripts regardless of testset contents
    tools_scripts = [
        "check_cuda.py",
        "check_all.py",
        "check_triton.py",
        "check_triton_gpu_all.py",
        "check_triton_vs_torch.py",
        "check_ascendc.py",
    ]
    tools_dir = context.settings.project_root / "llm_trans" / "tools"
    for script_name in tools_scripts:
        src_script = tools_dir / script_name
        dst_script = attempt_work_dir / script_name
        if src_script.exists():
            shutil.copy2(src_script, dst_script)
            logger.debug(f"Copied tool script {src_script} -> {dst_script}")
        else:
            logger.debug(f"Tool script {src_script} not found; skipping")

    # Read source code based on direction
    if direction == "cu2tri":
        source_file_path = attempt_work_dir / settings.dir_cuda / "kernel.cu"
        if not source_file_path.exists():
            logger.error(f"CUDA file not found: {source_file_path}")
            timing_stats.error = f"CUDA file missing: {source_file_path}"
            return False, None, timing_stats
        source_code = source_file_path.read_text()
    elif direction == "cu2asc":
        source_file_path = attempt_work_dir / settings.dir_cuda / "kernel.cu"
        if not source_file_path.exists():
            logger.error(f"CUDA file not found: {source_file_path}")
            timing_stats.error = f"CUDA file missing: {source_file_path}"
            return False, None, timing_stats
        source_code = source_file_path.read_text()
    elif direction == "tri2cute":
        source_file_path = attempt_work_dir / settings.dir_triton / "kernel.py"
        if not source_file_path.exists():
            logger.error(f"Triton file not found: {source_file_path}")
            timing_stats.error = f"Triton file missing: {source_file_path}"
            return False, None, timing_stats
        source_code = source_file_path.read_text()
    else:
        logger.error(f"Unsupported direction: {direction}")
        timing_stats.error = f"Unsupported direction: {direction}"
        return False, None, timing_stats

    history_manager = AttemptHistoryManager(
        attempt_work_dir=attempt_work_dir,
        attempt_number=attempt_number,
        case_type=case_type,
        case_name=case_name,
        resume=settings.resume_conversation,
        logger=logger,
    )

    # Select appropriate prompts based on translation direction
    if direction == "cu2tri":
        system_prompt = (
            "You are a professional GPU computing optimization expert, proficient in CUDA and Triton programming. "
            "You help convert CUDA kernels to Triton kernels while maintaining correctness and performance."
        )
        initial_user_prompt = cuda2triton_prompt.format(cuda_code=source_code)
    elif direction == "cu2asc":
        system_prompt = (
            "You are a professional NPU kernel engineer proficient in CUDA and Ascend C. "
            "Convert CUDA kernels into Ascend C kernels for NPU execution while preserving functional correctness."
        )
        initial_user_prompt = cuda2ascendc_prompt.format(cuda_code=source_code)
    elif direction == "tri2cute":
        system_prompt = (
            "You are a professional GPU computing optimization expert, proficient in Triton and CUTE/CUTLASS programming. "
            "You help convert Triton kernels to CUTE (CUTLASS 3.x/4.x) C++ kernels while maintaining correctness and performance."
        )
        initial_user_prompt = get_triton2cute_prompt(
            triton_code=source_code,
            use_simple=False,
            target_gpu=settings.target_gpu,
        )
    elif direction == 'tri2cutedsl':
        system_prompt = (
            "You are a professional GPU computing optimization expert, proficient in Triton and CUTE DSL programming. "
            "You help convert Triton kernels to CUTE (CUTLASS 4.x) Python kernels while maintaining correctness and performance."
        )
        initial_user_prompt = get_triton2cute_prompt(
            triton_code=source_code,
            use_simple=False,
            target_gpu=settings.target_gpu,
        )
    else:
        logger.error(f"Unsupported direction: {direction}")
        timing_stats.error = f"Unsupported direction: {direction}"
        return False, None, timing_stats
    
    history_manager.ensure_system_prompt(system_prompt)
    history_manager.ensure_initial_user_prompt(initial_user_prompt)

    conversation_history = history_manager.conversation

    assistant_messages = [msg for msg in conversation_history if msg.get("role") == "assistant"]
    initial_generation_needed = not assistant_messages

    resp_content = None
    actual_model_used = None

    if not initial_generation_needed:
        resp_content = assistant_messages[-1].get("content", "") if assistant_messages else None
        actual_model_used = model_name
        logger.info("Reusing assistant response from existing history (round 1)")

    if initial_generation_needed:
        retry_count = 0
        round_id = 1
        round_entry = RoundRecord(round=round_id, retry_limit=args.max_retries)
        round_retry_call_ms = 0.0
        round_retry_wait_ms = 0.0
        round_start_wall = None
        initial_stage = "initial_llm_generation"

        while retry_count <= args.max_retries:
            retry_index = retry_count
            retry_wall_start = now_timestamp()
            if round_start_wall is None:
                round_start_wall = retry_wall_start

            resp_content = None
            usage_dict = None
            extra_info = None

            api_params = None
            api_params_exc = None
            try:
                # Extract temperature from settings if available
                temperature = getattr(args, 'temperature', None)
                extra_kwargs = {}
                if temperature is not None:
                    extra_kwargs['temperature'] = temperature

                api_params = get_api_param(conversation_history, model_name, **extra_kwargs)
            except Exception as exc:
                    api_params_exc = exc

            context_meta = {
                "wall_start": ensure_timezone(retry_wall_start).isoformat(),
            }
            if round_start_wall:
                context_meta["round_start"] = ensure_timezone(round_start_wall).isoformat()
            if api_params_exc is not None:
                context_meta["api_param_error"] = str(api_params_exc)

            await log_retry_event(
                context,
                attempt_work_dir,
                event_type="attempt_start",
                stage=initial_stage,
                case_type=case_type,
                case_name=case_name,
                attempt_number=attempt_number,
                round_id=round_id,
                retry_index=retry_index,
                extra={
                    "model": model_name,
                    "attempt_started_at": ensure_timezone(retry_wall_start).isoformat(),
                    "retry_count": retry_count,
                    "max_retries": args.max_retries,
                },
                context_meta=build_retry_context(
                    conversation_history=conversation_history,
                    retry_count=retry_index,
                    max_retries=args.max_retries,
                    api_params=api_params if api_params_exc is None else None,
                    additional_meta=context_meta,
                ),
            )

            if api_params_exc is not None:
                raise api_params_exc

            try:
                with trans_timer.time("llm_call"):
                    result = await async_openai_llm_call(
                        async_client, api_params, logger=logger
                    )
                resp_reasoning_content, resp_content, actual_model_used, usage_dict, extra_info = result

                if not actual_model_used:
                    actual_model_used = model_name

                if actual_model_used != model_name:
                    logger.info(f"🤖 Model used: {actual_model_used} (requested: {model_name})")
                else:
                    logger.debug(f"🤖 Model used: {actual_model_used}")

                # Log temperature if used
                temperature = getattr(args, 'temperature', None)
                if temperature is not None:
                    logger.info(f"🌡️  Temperature: {temperature}")

                history_manager.add_assistant_message(
                    resp_reasoning_content,
                    resp_content,
                    round_id=round_id,
                    retry_index=retry_index,
                    metadata={"model_used": actual_model_used, "usage": usage_dict or {}},
                )

                await save_llm_conversation(
                    context=context,
                    test_work_dir=attempt_work_dir,
                    attempt_number=attempt_number,
                    round_id=round_id,
                    retry_index=retry_index,
                    messages=conversation_history,
                    full_response=resp_content,
                    model_used=actual_model_used,
                    reasoning_content=resp_reasoning_content,
                )

                if extra_info and extra_info.get("native_tokens_reasoning") is not None:
                    usage_dict = usage_dict or {}
                    usage_dict["reasoning_tokens"] = extra_info.get("native_tokens_reasoning")

                retry_wall_end = now_timestamp()
                retry_duration_ms = trans_timer.last_duration_ms("llm_call") or 0.0

                retry_record = RetryRecord(
                    retry_index=retry_index,
                    start_time=ensure_timezone(retry_wall_start).isoformat(),
                    end_time=ensure_timezone(retry_wall_end).isoformat(),
                    duration_ms=round(retry_duration_ms, 3),
                    success=True,
                    error=None,
                    usage=usage_dict,
                    extra_info=extra_info,
                )
                round_entry.add_retry(retry_record)

                await log_retry_event(
                    context,
                    attempt_work_dir,
                    event_type="attempt_finish",
                    stage=initial_stage,
                    case_type=case_type,
                    case_name=case_name,
                    attempt_number=attempt_number,
                    round_id=round_id,
                    retry_index=retry_index,
                    extra=retry_record.to_dict(),
                    context_meta=build_retry_context(
                        conversation_history=conversation_history,
                        retry_count=retry_index,
                        max_retries=args.max_retries,
                        api_params=api_params,
                        additional_meta={
                            **context_meta,
                            "status": "success",
                            "model_used": actual_model_used,
                        },
                    ),
                )

                if timing_stats is not None:
                    timing_stats.add_llm_time(retry_duration_ms)
                round_retry_call_ms += retry_duration_ms
                break

            except Exception as exc:
                retry_wall_end = now_timestamp()
                retry_duration_ms = trans_timer.last_duration_ms("llm_call") or 0.0
                error_msg = str(exc)

                retry_record = RetryRecord(
                    retry_index=retry_index,
                    start_time=ensure_timezone(retry_wall_start).isoformat(),
                    end_time=ensure_timezone(retry_wall_end).isoformat(),
                    duration_ms=round(retry_duration_ms, 3),
                    success=False,
                    error=error_msg,
                    usage=None,
                    extra_info=None,
                )
                round_entry.add_retry(retry_record)
                error_meta = {
                    **context_meta,
                    "error": error_msg,
                    "exception_type": type(exc).__name__,
                }
                if resp_content:
                    error_meta["response"] = resp_content
                    error_meta["response_char_count"] = len(resp_content)

                history_manager.add_error_event(
                    stage=initial_stage,
                    round_id=round_id,
                    retry_index=retry_index,
                    error=error_msg,
                    metadata={"exception_type": type(exc).__name__},
                )

                await log_retry_event(
                    context,
                    attempt_work_dir,
                    event_type="attempt_finish",
                    stage=initial_stage,
                    case_type=case_type,
                    case_name=case_name,
                    attempt_number=attempt_number,
                    round_id=round_id,
                    retry_index=retry_index,
                    extra=retry_record.to_dict(),
                    context_meta=build_retry_context(
                        conversation_history=conversation_history,
                        retry_count=retry_index,
                        max_retries=args.max_retries,
                        api_params=api_params,
                        additional_meta={**error_meta, "status": "error"},
                    ),
                )

                timing_stats.add_llm_retry_time(retry_duration_ms)
                round_retry_call_ms += retry_duration_ms

                if is_retryable_error(error_msg) and retry_count < args.max_retries:
                    if getattr(args, "rotate_endpoints", False):
                        ep = context.model.rotate_next()
                        if ep is not None:
                            logger.info(
                                f"🔁 Switched endpoint to '{ep.name}' at {ep.base_url or 'default'} for next retry"
                            )
                    retry_count += 1
                    wait_time = args.retry_wait
                    logger.warning(
                        f"🔄 API overload detected (retry {retry_count}/{args.max_retries}): {error_msg}"
                    )
                    if wait_time > 0:
                        logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                        with trans_timer.time("retry_wait"):
                            for remaining in range(wait_time, 0, -1):
                                if remaining % 10 == 0 or remaining <= 5:
                                    logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                                await asyncio.sleep(1)
                        waited_ms = trans_timer.last_duration_ms("retry_wait") or 0.0
                        timing_stats.add_llm_retry_wait(waited_ms)
                        round_retry_wait_ms += waited_ms
                        await log_retry_event(
                            context,
                            attempt_work_dir,
                            event_type="retry_wait",
                            stage=initial_stage,
                            case_type=case_type,
                            case_name=case_name,
                            attempt_number=attempt_number,
                            round_id=round_id,
                            retry_index=retry_count,
                            extra={
                                "reason": "retryable_error",
                                "error": error_msg,
                                "wait_seconds": wait_time,
                                "wait_ms": round(waited_ms, 3),
                            },
                            context_meta=build_retry_context(
                                conversation_history=conversation_history,
                                retry_count=retry_count,
                                max_retries=args.max_retries,
                                api_params=api_params,
                                additional_meta={
                                    **error_meta,
                                    "status": "retry_wait_error",
                                    "wait_seconds": wait_time,
                                    "wait_ms": round(waited_ms, 3),
                                },
                            ),
                        )
                    logger.info(f"🔄 Retrying API call (attempt {retry_count + 1})...")
                    continue

            if retry_count >= args.max_retries:
                logger.error(f"❌ Max retries ({args.max_retries}) exceeded for initial generation")
            logger.error(f"Failed to generate initial Triton code: {error_msg}", exc_info=True)
            timing_stats.error = error_msg
            return False, None, timing_stats

    if initial_generation_needed:
        if resp_content is None:
            logger.error("Failed to generate initial Triton code after all retries")
            timing_stats.error = "Initial generation failed"
            return False, None, timing_stats

        round_entry.status = "success"
        timing_stats.add_llm_round(round_entry)
    else:
        if resp_content is None:
            logger.error("No assistant message available in history to resume from")
            timing_stats.error = "Missing assistant response for resume"
            return False, None, timing_stats

    # Save generated code to appropriate directory based on direction
    generated_code = get_last_code_block(resp_content)
    
    if direction == "cu2tri":
        target_dir = attempt_work_dir / settings.dir_triton
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "kernel.py"
        target_file.write_text(generated_code)
        logger.info("Triton code generated successfully")
    elif direction == "cu2asc":
        target_dir = attempt_work_dir / settings.dir_ascendc
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "kernel.cpp"
        target_file.write_text(generated_code)
        logger.info("Ascend C code generated successfully")
    elif direction == "tri2cute":
        target_dir = attempt_work_dir / settings.dir_cute
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "kernel.cu"
        target_file.write_text(generated_code)
        logger.info("CUTE code generated successfully")
    else:
        logger.error(f"Unsupported direction: {direction}")
        timing_stats.error = f"Unsupported direction: {direction}"
        return False, None, timing_stats
    
    # For backward compatibility
    triton_dir = attempt_work_dir / settings.dir_triton
    triton_dir.mkdir(parents=True, exist_ok=True)

    save_conversation_history(
        context,
        history_manager.conversation,
        attempt_work_dir,
        round_num=1,
        timestamp=context.settings.timestamp,
    )

    logger.info("Starting automated testing and fixing process...")

    success, rounds = await run_testing_loop(
        context,
        history_manager,
        attempt_work_dir,
        timing_stats,
        attempt_number=attempt_number,
        case_type=case_type,
        case_name=case_name,
    )

    case_end_wall = now_timestamp()
    case_wall_duration_ms = trans_timer.stop("case_total") or 0.0

    effective_processing_time_ms = timing_stats.total_llm_time_ms + timing_stats.total_test_time_ms
    total_retry_overhead_ms = timing_stats.llm_retry_time_ms + timing_stats.llm_retry_wait_time_ms
    other_overhead_ms = max(
        case_wall_duration_ms - (effective_processing_time_ms + total_retry_overhead_ms),
        0.0,
    )

    timing_stats.end_time = ensure_timezone(case_end_wall).isoformat()
    timing_stats.wall_clock_duration_ms = round(case_wall_duration_ms, 3)
    timing_stats.effective_duration_ms = round(effective_processing_time_ms, 3)
    timing_stats.retry_overhead_ms = round(total_retry_overhead_ms, 3)
    timing_stats.other_overhead_ms = round(other_overhead_ms, 3)
    timing_stats.success = success
    timing_stats.final_round = rounds
    timing_stats.timestamp = context.settings.timestamp
    timing_stats.model_name = model_name
    timing_stats.actual_model_used = actual_model_used if actual_model_used else model_name
    timing_stats.testset = args.testset
    timing_stats.use_nvgpu = settings.use_nvgpu
    timing_stats.concurrency = args.concurrency
    timing_stats.gpu_server = args.nvgpu_server if settings.use_nvgpu else None
    timing_stats.max_attempts = args.max_attempts
    timing_stats.attempt_policy = args.attempt_policy
    timing_stats.ms_format = args.ms_format
    timing_stats.work_dir = str(attempt_work_dir)

    # Merge aggregated timer metrics
    try:
        timers = trans_timer.as_dict()
        if timers:
            if timing_stats.aggregated_timers is None:
                timing_stats.aggregated_timers = {}
            for label, value in timers.items():
                timing_stats.aggregated_timers[label] = timing_stats.aggregated_timers.get(label, 0.0) + float(value)
    except Exception:
        pass

    await write_jsonl_log(context, timing_stats.to_dict())

    # Persist a copy in attempt logs directory for local analysis/resume
    try:
        logs_dir = attempt_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "attempt_result.json").write_text(
            json.dumps(timing_stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Best-effort local copy; do not break main flow
        pass

    logger.info(
        "⏱️  Timing: Wall=%0.1fms, Effective(LLM+Test)=%0.1fms (LLM=%0.1fms, Test=%0.1fms, RetryCalls=%0.1fms, RetryWait=%0.1fms, Other=%0.1fms)",
        case_wall_duration_ms,
        effective_processing_time_ms,
        timing_stats.total_llm_time_ms,
        timing_stats.total_test_time_ms,
        timing_stats.llm_retry_time_ms,
        timing_stats.llm_retry_wait_time_ms,
        other_overhead_ms,
    )

    # Print aggregated timers for visibility
    timers_display = trans_timer.as_dict()
    if timers_display:
        parts = [f"{label}={format_ms(value, args.ms_format)}" for label, value in sorted(timers_display.items())]
        logger.info("⏱️  Aggregated timers: %s", ", ".join(parts))

    final_model = timing_stats.actual_model_used or model_name
    if final_model != model_name:
        logger.info(f"🤖 Final model used: {final_model} (requested: {model_name})")

    return success, rounds, timing_stats


async def run_single_case_translation(
    context: RuntimeContext,
    case_type: str,
    case_name: str,
    attempt_semaphore: asyncio.Semaphore | None = None,
) -> tuple[bool, int | None, list[AttemptResult]]:
    logger = context.logger
    settings = context.settings
    args = context.args

    case_root_dir = settings.work_dir / case_name

    attempt_results: list[AttemptResult] = []
    success_any = False
    best_rounds: int | None = None

    async def execute_attempt(attempt_number: int) -> tuple[bool, int | None, AttemptTimingStats] | Exception:
        attempt_work_dir = case_root_dir / f"attempt_{attempt_number:02d}"
        try:
            if attempt_semaphore is not None:
                async with attempt_semaphore:
                    return await run_single_case_attempt(context, case_type, case_name, attempt_number, attempt_work_dir)
            return await run_single_case_attempt(context, case_type, case_name, attempt_number, attempt_work_dir)
        except Exception as exc:
            return exc

    async def handle_attempt(attempt_number: int, outcome: tuple | Exception):
        nonlocal success_any, best_rounds
        attempt_work_dir = case_root_dir / f"attempt_{attempt_number:02d}"
        if isinstance(outcome, Exception):
            logger.error(f"Attempt {attempt_number} for {case_type}/{case_name} raised: {outcome}")
            stats = AttemptTimingStats(
                case_type=case_type,
                case_name=case_name,
                attempt_number=attempt_number,
                start_time=ensure_timezone(now_timestamp()).isoformat(),
                success=False,
                final_round=None,
                error=str(outcome),
                work_dir=str(attempt_work_dir),
            )
            await write_jsonl_log(context, stats.to_dict())
            # Also persist a local copy in attempt logs directory
            try:
                logs_dir = attempt_work_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                (logs_dir / "attempt_result.json").write_text(
                    json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            attempt_results.append(
                AttemptResult(
                    attempt_number=attempt_number,
                    success=False,
                    rounds=None,
                    timing_stats=stats,
                )
            )
            return False, None

        success, rounds, stats = outcome
        attempt_results.append(
            AttemptResult(
                attempt_number=attempt_number,
                success=success,
                rounds=rounds,
                timing_stats=stats,
            )
        )
        if success:
            success_any = True
            if rounds is not None:
                best_rounds = rounds if best_rounds is None else min(best_rounds, rounds)
        return success, rounds

    if args.attempt_policy == "exhaustive" and args.max_attempts > 1:
        attempt_numbers = list(range(1, args.max_attempts + 1))
        attempt_tasks = [asyncio.create_task(execute_attempt(num)) for num in attempt_numbers]
        outcomes = await asyncio.gather(*attempt_tasks, return_exceptions=True)
        for attempt_number, outcome in zip(attempt_numbers, outcomes):
            await handle_attempt(attempt_number, outcome)
    else:
        for attempt_number in range(1, args.max_attempts + 1):
            outcome = await execute_attempt(attempt_number)
            success, rounds = await handle_attempt(attempt_number, outcome)
            if success and args.attempt_policy == "first_success":
                logger.info(
                    f"✅ Case {case_type}/{case_name} succeeded on attempt {attempt_number}/{args.max_attempts}"
                )
                break

    attempt_results.sort(key=lambda entry: entry.attempt_number)

    if success_any and best_rounds is None:
        best_rounds = next((res.rounds for res in attempt_results if res.success), None)
    if not success_any and attempt_results:
        best_rounds = attempt_results[-1].rounds

    return success_any, best_rounds, attempt_results


__all__ = ["run_single_case_attempt", "run_single_case_translation"]
