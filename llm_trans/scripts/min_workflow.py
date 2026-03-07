#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal one-shot workflow for debugging transpilation via OpenRouter (gpt-5-mini).

Features
- Single attempt, no retry
- Full timing (init/load_input/call_llm/write_output/total)
- Specify --input single file, --workdir, --outputs_dir
- Stream or non-stream mode
- Outputs: output.txt, metadata.json (and stream.log when --stream)

Dependencies
- pip install openai
- Environment: OPENROUTER_API_KEY

Examples
- Non-stream:
  python /data/apps/project/cu2tri/cu2til/llm_trans/min_workflow.py \
    --input /abs/path/in.py \
    --outputs_dir /abs/out \
    --system "You are a transpiler." \
    --instruction "Transpile Triton to CUTe."

- Stream:
  python /data/apps/project/cu2tri/cu2til/llm_trans/min_workflow.py \
    --input /abs/path/in.py \
    --outputs_dir /abs/out \
    --stream
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Ensure project root is importable: /data/apps/project/cu2tri
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import provider interfaces
from llm.providers.factory import get_provider  # type: ignore
from llm.providers.types import PlatformType  # type: ignore
from llm.providers.base import ChatRequest  # type: ignore
from server.common.timer import (  # type: ignore
    create_host_timer,
    TimerSample,
    perf_counter_timestamp_ns,
    ns_to_ms,
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_messages(system_prompt: Optional[str], instruction: str, file_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Keep formatting minimal; user message includes instruction + source
    content = f"{instruction}\n\n{file_text}"
    messages.append({"role": "user", "content": content})
    return messages


def write_metadata(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def call_llm_non_stream(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    provider = get_provider(platform_type=PlatformType.OPENROUTER)
    req = ChatRequest(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    resp = await provider.chat(req)
    return {
        "content": resp.content or "",
        "finish_reason": resp.finish_reason,
        "usage": resp.usage,
        "model": resp.model,
        "processing_time": resp.processing_time,
        "thoughts": getattr(resp, "thoughts", None),
    }


async def call_llm_stream(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: Optional[float],
    out_path: Path,
    stream_log_path: Path,
    echo_stdout: bool = True,
) -> Dict[str, Any]:
    provider = get_provider(platform_type=PlatformType.OPENROUTER)
    req = ChatRequest(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    chunk_count = 0
    t0_ns = perf_counter_timestamp_ns()
    ensure_dir(out_path.parent)
    # Write chunks incrementally
    with out_path.open("w", encoding="utf-8") as out_f, stream_log_path.open("w", encoding="utf-8") as log_f:
        async for ch in provider.stream_chat(req):
            if not ch or not getattr(ch, "content", None):
                continue
            text = ch.content
            out_f.write(text)
            out_f.flush()
            if echo_stdout:
                print(text, end="", flush=True)
            # Log chunk timing (relative seconds and length)
            entry = {
                "t_rel_ms": float(ns_to_ms(perf_counter_timestamp_ns() - t0_ns)),
                "len": len(text),
                "type": getattr(ch, "content_type", None),
                "finish": getattr(ch, "finish_reason", None),
            }
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            chunk_count += 1

    # Read back full content for metadata convenience (optional)
    full_text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    return {
        "content": full_text,
        "chunk_count": chunk_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal one-shot transpile workflow via OpenRouter")
    parser.add_argument("--input", required=True, help="Path to input source file")
    parser.add_argument("--outputs_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--workdir", default=None, help="Working directory; default: <outputs_dir>/work")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="Model name (OpenRouter format)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument(
        "--instruction",
        default="Transpile the given code; keep formatting minimal.",
        help="High-level instruction for the model",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()

    # Timer with capture reporter
    samples: List[TimerSample] = []
    durations_ms: Dict[str, float] = {}

    def _capture(sample: TimerSample) -> None:
        samples.append(sample)
        durations_ms[sample.label] = sample.duration_ms

    timer = create_host_timer(reporter=_capture, enabled=True)

    # Resolve paths
    input_path = Path(args.input).expanduser().resolve()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else outputs_dir / "work"
    ensure_dir(outputs_dir)
    ensure_dir(workdir)

    # Prepare output files
    output_txt = outputs_dir / "output.txt"
    metadata_json = outputs_dir / "metadata.json"
    stream_log = outputs_dir / "stream.log"

    # Timers
    started_at = iso_now()

    error: Optional[str] = None
    exception_type: Optional[str] = None
    chunk_count: int = 0

    with timer.time("total"):
        # Stage: init (lightweight setup only)
        with timer.time("init"):
            pass  # keep stage for symmetry

        # Stage: load_input
        with timer.time("load_input"):
            if not input_path.exists() or not input_path.is_file():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            file_text = read_text_file(input_path)
            input_size = input_path.stat().st_size
            input_sha256 = file_sha256(input_path)

        # Build messages
        messages = build_messages(args.system, args.instruction, file_text)

        # Stage: call_llm
        result: Dict[str, Any] = {}
        try:
            with timer.time("call_llm"):
                if args.stream:
                    result = await call_llm_stream(
                        model=args.model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        out_path=output_txt,
                        stream_log_path=stream_log,
                        echo_stdout=True,
                    )
                    chunk_count = int(result.get("chunk_count", 0))
                else:
                    result = await call_llm_non_stream(
                        model=args.model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
        except Exception as e:  # Capture for metadata
            error = str(e)
            exception_type = e.__class__.__name__

        # Stage: write_output (non-stream path writes here)
        with timer.time("write_output"):
            if not args.stream and error is None:
                ensure_dir(output_txt.parent)
                output_txt.write_text(result.get("content", ""), encoding="utf-8")

    # Compose metadata
    finished_at = iso_now()
    # Compose metadata
    finished_at = iso_now()
    # durations in ms collected by reporter
    meta: Dict[str, Any] = {
        "attempt": 1,
        "retries": 0,
        "status": "success" if error is None else "error",
        "provider": {
            "platform": "openrouter",
            "model": args.model,
            "stream": bool(args.stream),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "inputs": {
            "input_file": str(input_path),
            "size": int(input_size),
            "sha256": input_sha256,
        },
        "outputs": {
            "output_file": str(output_txt),
            **({"stream_log": str(stream_log)} if args.stream else {}),
        },
        "durations_ms": {
            k: float(v) for k, v in durations_ms.items()
        },
        "durations": {
            k: float(v) / 1000.0 for k, v in durations_ms.items()
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "chunk_count": int(chunk_count),
    }
    if error is not None:
        meta["error"] = {"message": error, "type": exception_type}

    write_metadata(metadata_json, meta)

    # Return non-zero exit when error
    if error is not None:
        # Also echo a concise message to stderr
        sys.stderr.write(f"Error: {error}\n")
        return 2
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(async_main())
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


