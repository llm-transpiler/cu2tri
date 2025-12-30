from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..cli import prepare_context
from ..services.perf import run as run_perf_service


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NVGPU performance tests for existing cu2tri runs "
        "based on config/perf_runs.yaml.",
    )
    parser.add_argument(
        "--direction",
        choices=["cu2tri", "tri2cute"],
        default="cu2tri",
        help="Translation direction to use when resolving perf_runs.yaml (default: cu2tri).",
    )
    parser.add_argument(
        "--testset",
        default="xpiler",
        help="Testset key to use when resolving perf_runs.yaml (default: xpiler).",
    )
    parser.add_argument(
        "--model",
        default="gpt_oss_20b",
        help="Model name/key used only to build a RuntimeContext (perf does not call the LLM).",
    )
    parser.add_argument(
        "--nvgpu-server",
        default="http://localhost:8080",
        help="NVGPU server URL.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Reuse the main CLI's context builder so logging/settings are consistent.
    ctx = prepare_context(
        [
            "--direction",
            args.direction,
            "--testset",
            args.testset,
            "--model",
            args.model,
            "--use-nvgpu",
            "--nvgpu-server",
            args.nvgpu_server,
        ]
    )

    # 在 perf/logs 下实时记录本次批量 perf 的聚合日志（与控制台同步）
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log = log_dir / f"perf_{session_stamp}.log"
    latest_log = log_dir / "perf.log"

    logger = ctx.logger
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-7s - %(message)s"
    )

    fh_session = logging.FileHandler(session_log, mode="w", encoding="utf-8")
    fh_session.setLevel(logging.DEBUG)
    fh_session.setFormatter(formatter)

    fh_latest = logging.FileHandler(latest_log, mode="w", encoding="utf-8")
    fh_latest.setLevel(logging.DEBUG)
    fh_latest.setFormatter(formatter)

    logger.addHandler(fh_session)
    logger.addHandler(fh_latest)

    try:
        summary = asyncio.run(run_perf_service(ctx))
    finally:
        # 移除临时 handler，避免影响后续其它运行
        logger.removeHandler(fh_session)
        logger.removeHandler(fh_latest)
        fh_session.close()
        fh_latest.close()

    # 打印到标准输出，方便脚本调用
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


