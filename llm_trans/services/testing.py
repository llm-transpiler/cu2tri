from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from ..utils.trans_timer import TransTimer
from server.common.timezone import (
    ensure_timezone,
    normalize_timestamp_iso,
    now_timestamp,
    parse_timestamp,
)
from server.common.task_refs import format_task_ref

from llm_trans.prompts.cuda2triton import feedback_prompt as cuda2triton_feedback
from llm_trans.prompts.cuda2ascendc import feedback_prompt as cuda2ascendc_feedback

from ..clients import (
    NVGPU_AVAILABLE,
    NVGPUClient,
    async_openai_llm_call,
)
from ..core.runtime import RuntimeContext
from ..data.models import AttemptTimingStats, RetryRecord, RoundRecord, TestRoundRecord
from ..services.retry import build_retry_context, is_retryable_error, log_retry_event
from ..services.conversation import get_last_code_block, save_conversation_history, save_llm_conversation
from ..services.history import AttemptHistoryManager
from ..utils.formatting import format_ms


async def run_test_round(
    context: RuntimeContext,
    round_num: int,
    test_work_dir: Path,
    timing_stats: AttemptTimingStats | None = None,
    *,
    task_label: str | None = None,
):
    logger = context.logger
    settings = context.settings

    logger.debug(f"Running test round {round_num}...")
    
    # Route to appropriate testing method based on direction
    direction = settings.direction
    
    if direction == "tri2cute":
        # Triton to CUTE: test CUTE code
        from .testing_cute import run_test_round_cute
        return await run_test_round_cute(context, round_num, test_work_dir, timing_stats)
    else:
        if direction == "cu2tri":
            kernel_dir = settings.dir_triton
            kernel_name = "kernel.py"
            log_prefix = "triton"
        elif direction == "cu2asc":
            kernel_dir = settings.dir_ascendc
            kernel_name = "kernel.cpp"
            log_prefix = "ascendc"
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        kernel_path = test_work_dir / kernel_dir / kernel_name
        backup_path = test_work_dir / kernel_dir / f"kernel_v{round_num}{Path(kernel_name).suffix}"
        shutil.copy(kernel_path, backup_path)
        logger.debug(f"Backed up kernel to {backup_path.name}")

        logs_dir = test_work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{log_prefix}_test_round_{round_num}.log"

        if not (settings.use_nvgpu and context.nvgpu_available):
            raise ValueError("NVGPU is not available")

        return await run_test_round_nvgpu(
            context,
            round_num,
            test_work_dir,
            log_file,
            timing_stats,
            task_label=task_label,
        )


async def run_test_round_nvgpu(
    context: RuntimeContext,
    round_num: int,
    test_work_dir: Path,
    log_file: Path,
    timing_stats: AttemptTimingStats | None = None,
    *,
    task_label: str | None = None,
):
    logger = context.logger
    args = context.args

    try:
        nvgpu_client = NVGPUClient(args.nvgpu_server)
        if not nvgpu_client.health_check():
            logger.error(f"NVGPU server at {args.nvgpu_server} is not responding")
            return False, "", "NVGPU server not responding", log_file

        logger.debug(f"Connected to NVGPU server at {args.nvgpu_server}")

        direction = context.settings.direction
        if direction == "cu2asc":
            script_name = "check_ascendc.py"
        else:
            script_name = f"check_triton{context.settings.check_suffix}.py"
        script_path = str((test_work_dir / script_name).absolute())
        # For internal check scripts: by default禁用内部perf，只有在显式开启enable_perf时才不传--no-perf
        task_args = ["--no-perf"] if not getattr(args, "enable_perf", False) else []

        logger.info(f"Submitting task to NPU server (NPU: {args.nvgpu_gpu or 'auto'})")
        project_root = str(context.settings.project_root)
        merged_pythonpath = project_root
        if os.getenv("PYTHONPATH"):
            merged_pythonpath = f"{project_root}:{os.getenv('PYTHONPATH')}"

        task_env = {
            "PROJECT_ROOT": project_root,
            "PYTHONPATH": merged_pythonpath,
        }

        task_id = nvgpu_client.submit_task_in_script_dir(
            script_path=script_path,
            task_type="functional",
            task_label=task_label,
            args=task_args,
            gpu_id=args.nvgpu_gpu,
            env=task_env,
        )
        logger.info("Task submitted: %s", format_task_ref(task_id, include_label=False))
        last_status = None

        def _format_status_progress(result_obj):
            parts = []
            if result_obj.total_duration_ms is not None:
                parts.append(f"total={format_ms(result_obj.total_duration_ms, args.ms_format)}")
            else:
                if result_obj.pending_duration_ms is not None:
                    parts.append(f"pending={format_ms(result_obj.pending_duration_ms, args.ms_format)}")
                if result_obj.queue_duration_ms is not None:
                    parts.append(f"queue={format_ms(result_obj.queue_duration_ms, args.ms_format)}")
                if result_obj.waiting_duration_ms is not None:
                    parts.append(f"waiting={format_ms(result_obj.waiting_duration_ms, args.ms_format)}")
                if result_obj.running_duration_ms is not None:
                    parts.append(f"running={format_ms(result_obj.running_duration_ms, args.ms_format)}")
            return ", ".join(parts)

        while True:
            result = nvgpu_client.get_task(task_id)
            current_status = result.status

            if current_status != last_status:
                progress_text = _format_status_progress(result)
                task_ref = format_task_ref(result, include_label=True)
                if progress_text:
                    logger.info("TASK %s: %s (%s)", task_ref, current_status, progress_text)
                else:
                    logger.info("TASK %s: %s", task_ref, current_status)
                last_status = current_status

            if current_status in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(2)

        submit_ts_raw = getattr(result, "submit_timestamp", getattr(result, "submit_time", None))
        queued_ts_raw = getattr(result, "queued_timestamp", None)
        start_ts_raw = getattr(result, "start_timestamp", getattr(result, "start_time", None))
        end_ts_raw = getattr(result, "end_timestamp", getattr(result, "end_time", None))

        start_dt = parse_timestamp(start_ts_raw)
        end_dt = parse_timestamp(end_ts_raw)

        running_ms = getattr(result, "running_duration_ms", None)
        waiting_ms = getattr(result, "waiting_duration_ms", None)
        queue_ms = getattr(result, "queue_duration_ms", None)
        pending_ms = getattr(result, "pending_duration_ms", None)
        total_ms = getattr(result, "total_duration_ms", None)

        execution_time_ms = running_ms
        if execution_time_ms is None and start_dt and end_dt:
            execution_time_ms = (end_dt - start_dt).total_seconds() * 1000.0

        waiting_time_ms = waiting_ms
        if waiting_time_ms is None and execution_time_ms is not None and total_ms is not None:
            waiting_time_ms = max(total_ms - execution_time_ms, 0.0)

        queue_time_ms = queue_ms if queue_ms is not None else None
        pending_time_ms = pending_ms if pending_ms is not None else None
        total_server_time_ms = total_ms if total_ms is not None else None

        start_iso = normalize_timestamp_iso(start_ts_raw)
        end_iso = normalize_timestamp_iso(end_ts_raw)
        submit_iso = normalize_timestamp_iso(submit_ts_raw)
        queued_iso = normalize_timestamp_iso(queued_ts_raw)

        assigned_gpu = getattr(result, "gpu_id", getattr(result, "assigned_gpu", None))

        if timing_stats is not None:
            record = TestRoundRecord(
                round=round_num,
                success=result.status == "completed" and result.exit_code == 0,
                execution_mode="nvgpu",
                start_time=start_iso,
                end_time=end_iso,
                duration_ms=round(execution_time_ms, 3) if execution_time_ms is not None else None,
                waiting_ms=round(waiting_time_ms, 3) if waiting_time_ms is not None else None,
                queue_ms=round(queue_time_ms, 3) if queue_time_ms is not None else None,
                pending_ms=round(pending_time_ms, 3) if pending_time_ms is not None else None,
                total_server_ms=round(total_server_time_ms, 3) if total_server_time_ms is not None else None,
                gpu_id=assigned_gpu if assigned_gpu is not None else (args.nvgpu_gpu or "auto"),
                status=result.status,
                exit_code=getattr(result, "exit_code", None),
            )
            timing_stats.add_test_round(record)
            if execution_time_ms is not None:
                timing_stats.add_test_time(execution_time_ms)

        display_total_ms = total_ms if total_ms is not None else None

        timing_parts = []
        if pending_time_ms is not None:
            timing_parts.append(f"Pending: {format_ms(pending_time_ms, args.ms_format)}")
        if queue_time_ms is not None:
            timing_parts.append(f"Queue: {format_ms(queue_time_ms, args.ms_format)}")
        if waiting_time_ms is not None:
            timing_parts.append(f"Waiting: {format_ms(waiting_time_ms, args.ms_format)}")
        if execution_time_ms is not None:
            timing_parts.append(f"Execution: {format_ms(execution_time_ms, args.ms_format)}")
        if total_server_time_ms is not None:
            timing_parts.append(f"Total: {format_ms(total_server_time_ms, args.ms_format)}")
        if timing_parts:
            logger.info("  └─ " + ", ".join(timing_parts))

        try:
            stdout_content = nvgpu_client.get_full_task_log(task_id, log_type="stdout")
            logger.debug(f"Retrieved stdout ({len(stdout_content)} bytes)")
        except Exception as log_exc:
            logger.warning(f"Failed to fetch stdout for task {task_id}: {log_exc}")
            stdout_content = f"Error fetching stdout: {log_exc}"

        try:
            stderr_content = nvgpu_client.get_full_task_log(task_id, log_type="stderr")
            logger.debug(f"Retrieved stderr ({len(stderr_content)} bytes)")
        except Exception as log_exc:
            logger.warning(f"Failed to fetch stderr for task {task_id}: {log_exc}")
            stderr_content = f"Error fetching stderr: {log_exc}"

        with open(log_file, "w") as fp:
            fp.write(f"=== Test Round {round_num} (NVGPU) ===\n")
            try:
                fp.write(f"TASK: {format_task_ref(result, include_label=True)}\n")
            except Exception:
                pass
            fp.write(f"Task ID: {task_id}\n")
            fp.write(f"GPU: {assigned_gpu if assigned_gpu is not None else (args.nvgpu_gpu or 'auto-assigned')}\n")
            fp.write(f"Status: {result.status}\n")
            fp.write(f"Exit code: {result.exit_code}\n")
            fp.write("\n=== Timing Information ===\n")

            timing_ms_entries: list[tuple[str, float]] = []
            if display_total_ms is not None:
                timing_ms_entries.append(("Total elapsed time (submit to finish)", display_total_ms))
            if total_server_time_ms is not None:
                timing_ms_entries.append(("Server reported total time", total_server_time_ms))
            if pending_time_ms is not None:
                timing_ms_entries.append(("Pending time (submit to queue)", pending_time_ms))
            if queue_time_ms is not None:
                timing_ms_entries.append(("Queue time (assign to start)", queue_time_ms))
            if waiting_time_ms is not None:
                timing_ms_entries.append(("Waiting time (queue)", waiting_time_ms))
            if execution_time_ms is not None:
                timing_ms_entries.append(("Execution time (actual run)", execution_time_ms))

            if timing_ms_entries:
                label_width = max(len(label) for label, _ in timing_ms_entries)
                value_width = max(len(format_ms(value, args.ms_format)) for _, value in timing_ms_entries)
                for label, value in timing_ms_entries:
                    value_str = format_ms(value, args.ms_format)
                    fp.write(f"{label:<{label_width}} : {value_str:>{value_width}}\n")

            timestamp_entries = []
            if submit_iso:
                timestamp_entries.append(("Submit time", submit_iso))
            if queued_iso:
                timestamp_entries.append(("Queued time", queued_iso))
            if start_iso:
                timestamp_entries.append(("Start time", start_iso))
            if end_iso:
                timestamp_entries.append(("End time", end_iso))

            if timestamp_entries:
                label_width_ts = max(len(label) for label, _ in timestamp_entries)
                value_width_ts = max(len(value) for _, value in timestamp_entries)
                for label, value in timestamp_entries:
                    fp.write(f"{label:<{label_width_ts}} : {value:>{value_width_ts}}\n")

            fp.write(f"\nServer log: {result.log_file}\n\n")
            fp.write("=== STDOUT ===\n")
            fp.write(stdout_content)
            fp.write("\n=== STDERR ===\n")
            fp.write(stderr_content)
            fp.write("\n=== END OF LOG ===\n")

        logger.debug(f"Test output saved to {log_file}")

        success = result.status == "completed" and result.exit_code == 0 and "PASSED" in stdout_content
        return success, stdout_content, stderr_content, log_file

    except Exception as exc:  # pragma: no cover
        error_msg = f"NVGPU execution failed: {exc}"
        logger.error(error_msg, exc_info=True)
        with open(log_file, "w") as fp:
            fp.write(f"=== Test Round {round_num} ===\n")
            fp.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg, log_file


async def get_feedback_from_llm(
    context: RuntimeContext,
    history_manager: AttemptHistoryManager,
    round_num: int,
    error_output: str,
    stderr_output: str,
    test_work_dir: Path,
    *,
    case_type: str,
    case_name: str,
    timing_stats: AttemptTimingStats | None = None,
    attempt_number: int = 1,
) -> str | None:
    logger = context.logger
    args = context.args
    model_name = context.model.model_name

    # Ensure async_client is defined before any use to avoid UnboundLocalError
    async_client = context.model.async_client
    if async_client is None:
        logger.error("Async client is not configured")
        return None

    conversation_history = history_manager.conversation
    settings = context.settings
    direction = settings.direction

    error_info = (
        f"Round {round_num} Test Output:\n{error_output}\n\nStderr:\n{stderr_output}"
    )
    traceback_info = stderr_output if stderr_output else "No traceback available"
    
    # Select appropriate feedback prompt based on direction
    if direction == "tri2cute":
        # Get the current CUTE code for feedback
        cute_file = test_work_dir / settings.dir_cute / "kernel.cu"
        current_code = cute_file.read_text() if cute_file.exists() else "Code not found"
        prompt = get_triton2cute_feedback(error_output=error_output, current_code=current_code)
    elif direction == "cu2asc":
        prompt = cuda2ascendc_feedback.format(error_info=error_info, traceback_info=traceback_info)
    elif direction == "cu2tri":
        # cu2tri
        prompt = cuda2triton_feedback.format(error_info=error_info, traceback_info=traceback_info)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    feedback_round_id = round_num + 1

    if not conversation_history or conversation_history[-1].get("role") != "user":
        history_manager.add_user_message(
            prompt,
            round_id=feedback_round_id,
            retry_index=0,
            metadata={"stage": "feedback_prompt"},
        )

    retry_count = 0
    fixed_code_response = None
    feedback_model_used = None
    round_entry = RoundRecord(round=feedback_round_id, retry_limit=args.max_retries)
    round_start_wall = None
    feedback_stage = "feedback_llm_generation"

    while retry_count <= args.max_retries:
        retry_index = retry_count
        trans_timer = TransTimer()
        retry_wall_start = now_timestamp()
        if round_start_wall is None:
            round_start_wall = retry_wall_start

        fixed_code_response = None
        usage_dict = None
        extra_info = None

        api_params = None
        api_params_exc = None
        try:
            # Build API params against the currently selected endpoint
            get_api_param = context.model.get_api_param

            # Extract temperature from settings if available
            temperature = getattr(context.args, 'temperature', None)
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
            test_work_dir,
            event_type="attempt_start",
            stage=feedback_stage,
            case_type=case_type,
            case_name=case_name,
            attempt_number=attempt_number,
            round_id=feedback_round_id,
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
            # Use currently selected endpoint's async client
            async_client = context.model.async_client
            with trans_timer.time("feedback_call"):
                fixed_reasoning_content_response, fixed_code_response, feedback_model_used, usage_dict, extra_info = await async_openai_llm_call(
                    async_client, api_params, logger=logger
                )

            if not feedback_model_used:
                feedback_model_used = model_name

            if feedback_model_used != model_name:
                logger.info(
                    f"🤖 Feedback model used: {feedback_model_used} (requested: {model_name})"
                )
            else:
                logger.debug(f"🤖 Feedback model used: {feedback_model_used}")

            history_manager.add_assistant_message(
                fixed_reasoning_content_response,
                fixed_code_response,
                round_id=feedback_round_id,
                retry_index=retry_index,
                metadata={"model_used": feedback_model_used, "usage": usage_dict or {}},
            )

            await save_llm_conversation(
                context=context,
                test_work_dir=test_work_dir,
                attempt_number=attempt_number,
                round_id=feedback_round_id,
                retry_index=retry_index,
                messages=conversation_history,
                full_response=fixed_code_response,
                model_used=feedback_model_used,
                reasoning_content=fixed_reasoning_content_response,
            )

            if extra_info and extra_info.get("native_tokens_reasoning") is not None:
                usage_dict = usage_dict or {}
                usage_dict["reasoning_tokens"] = extra_info.get("native_tokens_reasoning")

            retry_wall_end = now_timestamp()
            retry_duration_ms = trans_timer.last_duration_ms("feedback_call") or 0.0

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
                test_work_dir,
                event_type="attempt_finish",
                stage=feedback_stage,
                case_type=case_type,
                case_name=case_name,
                attempt_number=attempt_number,
                round_id=feedback_round_id,
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
                        "model_used": feedback_model_used,
                    },
                ),
            )

            # Print feedback timers for this attempt
            timers_display = trans_timer.as_dict()
            if timers_display:
                parts = [f"{label}={format_ms(value, args.ms_format)}" for label, value in sorted(timers_display.items())]
                logger.info("⏱️  Feedback timers: %s", ", ".join(parts))

            if timing_stats is not None:
                timing_stats.add_llm_time(retry_duration_ms)

            round_entry.status = "success"
            if timing_stats is not None:
                timing_stats.add_llm_round(round_entry)
            return get_last_code_block(fixed_code_response)

        except Exception as exc:
            retry_wall_end = now_timestamp()
            retry_duration_ms = trans_timer.last_duration_ms("feedback_call") or 0.0
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
            if fixed_code_response:
                error_meta["response"] = fixed_code_response
                error_meta["response_char_count"] = len(fixed_code_response)

            history_manager.add_error_event(
                stage=feedback_stage,
                round_id=feedback_round_id,
                retry_index=retry_index,
                error=error_msg,
                metadata={"exception_type": type(exc).__name__},
            )

            await log_retry_event(
                context,
                test_work_dir,
                event_type="attempt_finish",
                stage=feedback_stage,
                case_type=case_type,
                case_name=case_name,
                attempt_number=attempt_number,
                round_id=feedback_round_id,
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

            # Print feedback timers for this attempt (error)
            timers_display = trans_timer.as_dict()
            if timers_display:
                parts = [f"{label}={format_ms(value, args.ms_format)}" for label, value in sorted(timers_display.items())]
                logger.info("⏱️  Feedback timers: %s", ", ".join(parts))

            if timing_stats is not None:
                timing_stats.add_llm_retry_time(retry_duration_ms)

            if is_retryable_error(error_msg) and retry_count < args.max_retries:
                retry_count += 1
                wait_time = args.retry_wait
                logger.warning(
                    f"🔄 API overload detected in feedback (retry {retry_count}/{args.max_retries}): {error_msg}"
                )
                if getattr(args, "rotate_endpoints", False):
                    ep = context.model.rotate_next()
                    if ep is not None:
                        logger.info(
                            f"🔁 Switched endpoint to '{ep.name}' at {ep.base_url or 'default'} for next retry"
                        )
                if wait_time > 0:
                    logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    with trans_timer.time("feedback_retry_wait"):
                        for remaining in range(wait_time, 0, -1):
                            if remaining % 10 == 0 or remaining <= 5:
                                logger.debug(f"⏱️  Retrying in {remaining} seconds...")
                            await asyncio.sleep(1)
                    waited_ms = trans_timer.last_duration_ms("feedback_retry_wait") or 0.0
                    if timing_stats is not None:
                        timing_stats.add_llm_retry_wait(waited_ms)
                    history_manager.add_error_event(
                        stage=feedback_stage,
                        round_id=feedback_round_id,
                        retry_index=retry_count,
                        error=f"Waiting {wait_time}s before retry",
                        metadata={"wait_ms": round(waited_ms, 3)},
                    )
                    await log_retry_event(
                        context,
                        test_work_dir,
                        event_type="retry_wait",
                        stage=feedback_stage,
                        case_type=case_type,
                        case_name=case_name,
                        attempt_number=attempt_number,
                        round_id=feedback_round_id,
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
                logger.info(f"🔄 Retrying feedback API call (attempt {retry_count + 1})...")
                continue

            if retry_count >= args.max_retries:
                logger.error(f"❌ Max retries ({args.max_retries}) exceeded for feedback round {round_num}")
            logger.error(f"Failed to get LLM feedback for round {round_num}: {error_msg}")
            return None

    logger.error("Failed to get LLM feedback after all retries")
    return None



async def run_testing_loop(
    context: RuntimeContext,
    history_manager: AttemptHistoryManager,
    test_work_dir: Path,
    timing_stats: AttemptTimingStats | None = None,
    *,
    attempt_number: int,
    case_type: str,
    case_name: str,
):
    logger = context.logger
    max_rounds = context.settings.max_rounds

    direction = context.settings.direction

    task_label = case_name
    if attempt_number:
        task_label += f"|at@{attempt_number}"
    for round_num in range(1, max_rounds + 1):
        success, stdout, stderr, log_file = await run_test_round(
            context,
            round_num,
            test_work_dir,
            timing_stats,
            task_label=f"{task_label}|r@{round_num}",
        )

        if success:
            # Infer target kernel label from direction string, e.g. "tri2cute" or "cu2tri"
            if "2" in direction:
                target_suffix = direction.split("2", 1)[1].lower()
            else:
                target_suffix = direction.lower()

            kernel_label_map = {
                "tri": "Triton",
                "triton": "Triton",
                "cute": "CuTe",
                "cu": "CUDA",
                "ascendc": "Ascend C",
            }
            kernel_label = kernel_label_map.get(target_suffix, target_suffix.capitalize())
            logger.info(f"✅ Test round {round_num} PASSED! {kernel_label} kernel is working correctly.")
            logger.info(f"Final results saved in {log_file}")
            # Optionally run performance testing after correctness (NVGPU only, gated by enable_perf)
            try:
                if not (context.settings.use_nvgpu and context.nvgpu_available):
                    logger.warning("Perf step skipped: NVGPU is not available")
                    return True, round_num
                elif getattr(context.args, "enable_perf", False):
                    from .perf import run_perf_nvgpu, record_perf_result

                    gpu_id = context.args.nvgpu_gpu
                    shape_tag = None  # optional: could be inferred from get_data if available

                    perf_data, server_times = await run_perf_nvgpu(
                        context,
                        test_work_dir,
                        gpu_id=gpu_id,
                        case_tag=case_name,
                        shape_tag=shape_tag,
                    )
                    await record_perf_result(context, test_work_dir, perf_data, server_times)
            except Exception as perf_exc:
                logger.warning(f"Perf step failed: {perf_exc}")
            return True, round_num

        logger.info(f"❌ Test round {round_num} FAILED.")
        logger.debug(f"Error details saved in {log_file}")

        if round_num < max_rounds:
            logger.info("🔧 Attempting to fix with LLM feedback...")

            fixed_code = await get_feedback_from_llm(
                context,
                history_manager,
                round_num,
                stdout,
                stderr,
                test_work_dir,
                case_type=case_type or "undefined",
                case_name=case_name or test_work_dir.name,
                timing_stats=timing_stats,
                attempt_number=attempt_number,
            )

            if fixed_code:
                # Save fixed code to appropriate location based on direction
                direction = context.settings.direction
                if direction == "cu2tri":
                    kernel_path = test_work_dir / context.settings.dir_triton / "kernel.py"
                elif direction == "cu2asc":
                    kernel_path = test_work_dir / context.settings.dir_ascendc / "kernel.cpp"
                elif direction == "tri2cute":
                    kernel_path = test_work_dir / context.settings.dir_cute / "kernel.cu"
                else:
                    logger.error(f"Unsupported direction: {direction}")
                    break
                
                with open(kernel_path, "w") as fp:
                    fp.write(fixed_code)
                logger.debug(f"Updated {kernel_path.name} with LLM feedback")
                # kernel_path.name : os.path.basename(str(kernel_path)) (kernel.cu or kernel.py)

                save_conversation_history(
                    context,
                    history_manager.conversation,
                    test_work_dir,
                    round_num=round_num + 1,
                    timestamp=context.settings.timestamp,
                )
            else:
                logger.error(f"Failed to get valid LLM feedback for round {round_num}")
                break
        else:
            logger.error(f"Maximum rounds ({max_rounds}) reached. Manual intervention required.")
            break

    return False, max_rounds


__all__ = [
    "run_test_round",
    "run_test_round_local",
    "run_test_round_nvgpu",
    "get_feedback_from_llm",
    "run_testing_loop",
]
