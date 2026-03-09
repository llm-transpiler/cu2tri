#!/usr/bin/env python3
"""Main entry point for NPU server."""
import argparse
import signal
import sys
from pathlib import Path

import uvicorn

from server.npu.api_server import app, init_app
from server.npu.config import LoadBalancingStrategy, NPUMode, config
from server.npu.logger import setup_logger
from server.npu.npu_config_loader import NPUConfigLoader
from server.npu.npu_manager import NPUManager
from server.npu.scheduler import Scheduler
from server.npu.task_queue import TaskQueue
from server.npu.task_runner import TaskRunner
from server.common.timezone import format_timestamp, set_default_timezone
from server.common.timer import monotonic_timestamp_ns, perf_counter_timestamp_ns

NPU_ROOT = Path(__file__).parent.resolve()

set_default_timezone(config.timezone)
timestamp = format_timestamp()
log_file = str(NPU_ROOT / "logs" / f"npu_server_{timestamp}.log")
history_log_file = str(NPU_ROOT / "logs" / "npu_server.log")
logger = setup_logger("main", log_file=log_file, history_log_file=history_log_file)

logger.info(
    "Timer setup: perf_counter_ns for durations (sample=%d), monotonic_ns for ordering (sample=%d)",
    perf_counter_timestamp_ns(),
    monotonic_timestamp_ns(),
)


class NPUServer:
    """Main NPU server class."""

    def __init__(
        self,
        npu_config_file: str | None = None,
        log_file_path: str | None = None,
        history_log_file_path: str | None = None,
    ):
        if log_file_path:
            self.log_file = log_file_path if Path(log_file_path).is_absolute() else str(NPU_ROOT / log_file_path)
        else:
            self.log_file = str(NPU_ROOT / "logs" / f"npu_server_{format_timestamp()}.log")

        if history_log_file_path:
            self.history_log_file = (
                history_log_file_path
                if Path(history_log_file_path).is_absolute()
                else str(NPU_ROOT / history_log_file_path)
            )
        else:
            self.history_log_file = str(NPU_ROOT / "logs" / "npu_server.log")

        self.logger = logger
        self.npu_config_loader = None

        if npu_config_file:
            self.logger.info("Loading NPU config from: %s", npu_config_file)
            self.npu_config_loader = NPUConfigLoader(npu_config_file)
            if self.npu_config_loader.load():
                import server.npu.npu_manager as npu_manager_mod
                import server.npu.task_runner as npu_task_runner_mod
                import server.nvgpu.task_runner as base_task_runner_mod

                npu_manager_mod.npu_config_loader = self.npu_config_loader
                npu_task_runner_mod.npu_config_loader = self.npu_config_loader
                # Keep base runner mapping in sync for inherited execution path.
                base_task_runner_mod.gpu_config_loader = self.npu_config_loader
                enabled_npus = self.npu_config_loader.get_enabled_npus()
                self.logger.info("NPU config loaded successfully: %d NPUs enabled", len(enabled_npus))
            else:
                self.logger.error("Failed to load NPU config from: %s", npu_config_file)

        self.npu_manager = NPUManager()
        self.task_queue = TaskQueue()
        self.task_runner = TaskRunner()
        self.npu_manager.set_dependencies(self.task_queue, self.task_runner)
        self.scheduler = Scheduler(self.npu_manager, self.task_queue, self.task_runner)

    def start(self):
        logger.info("=" * 60)
        logger.info("Starting NPU Server")
        logger.info("=" * 60)

        if self.npu_config_loader and self.npu_config_loader.should_auto_register():
            logger.info("Auto-registering NPUs from configuration...")
            for npu_config in self.npu_config_loader.get_enabled_npus():
                mode = NPUMode.EXCLUSIVE if npu_config.default_mode == "exclusive" else NPUMode.SHARED
                self.npu_manager.register_npu(
                    npu_id=npu_config.logical_id,
                    mode=mode,
                    memory_threshold=npu_config.memory_threshold,
                    max_concurrent_tasks=npu_config.max_concurrent_tasks,
                )
                logger.info(
                    "  Registered NPU %s: %s (mode=%s, max_tasks=%s)",
                    npu_config.logical_id,
                    npu_config.name,
                    mode,
                    npu_config.max_concurrent_tasks,
                )

        self.npu_manager.start_monitoring()
        self.npu_manager.clear_stale_running_tasks()
        self.scheduler.start()

        strategy_name = (
            config.load_balancing_strategy.value if config.load_balancing_strategy else LoadBalancingStrategy.FILL.value
        )
        logger.info("Load balancing strategy: %s", strategy_name)

        init_app(self.npu_manager, self.task_queue, self.scheduler, self.task_runner)
        logger.info("Server ready on http://%s:%s", config.host, config.port)
        logger.info("Registered NPUs: %s", [n.gpu_id for n in self.npu_manager.list_npus()])

        log_file_abs = str(Path(self.log_file).absolute())
        history_log_file_abs = str(Path(self.history_log_file).absolute())
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "server.npu.logger.TimezoneFormatter",
                    "format": "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
                "file": {"formatter": "default", "class": "logging.FileHandler", "filename": log_file_abs},
                "history_file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": history_log_file_abs,
                    "mode": "a",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default", "file", "history_file"], "level": "INFO"},
                "uvicorn.error": {
                    "handlers": ["default", "file", "history_file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["default", "file", "history_file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }

        uvicorn.run(app, host=config.host, port=config.port, log_config=log_config)

    def stop(self):
        logger.info("Shutting down NPU Server...")
        self.scheduler.stop()
        self.npu_manager.shutdown()
        logger.info("Server stopped")


def signal_handler(signum, frame):
    logger.info("Received signal %s, shutting down...", signum)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="NPU Server - NPU task scheduling service")
    parser.add_argument("--host", default=config.host, help="Server host")
    parser.add_argument("--port", type=int, default=config.port, help="Server port")
    parser.add_argument("--log-file", default=None, help="Log file path (default: auto-generated with timestamp)")
    parser.add_argument("--log-level", default=config.log_level, help="Log level")
    parser.add_argument("--npu-config", default=None, help="NPU resource configuration file (YAML)")
    parser.add_argument("--npus", type=int, nargs="+", help="NPU IDs to register at startup (overrides config file)")
    parser.add_argument("--npu-mode", choices=["exclusive", "shared"], help="Default NPU mode (when using --npus)")
    parser.add_argument("--memory-threshold", type=float, help="Default memory threshold (when using --npus)")
    parser.add_argument(
        "--lb-strategy",
        choices=["default"] + [s.value for s in LoadBalancingStrategy],
        default="round_robin",
        help="Load balancing strategy (default: legacy fill-first scheduling)",
    )

    args = parser.parse_args()

    global log_file, logger
    config.host = args.host
    config.port = args.port

    if args.log_file:
        log_file = args.log_file
        config.log_file = args.log_file
        logger.info("Using custom log file: %s", log_file)
    config.log_level = args.log_level
    if args.npu_mode:
        config.default_gpu_mode = NPUMode(args.npu_mode)
    if args.memory_threshold:
        config.default_memory_threshold = args.memory_threshold
    if args.lb_strategy == "default":
        config.load_balancing_strategy = None
    else:
        strategy = LoadBalancingStrategy(args.lb_strategy)
        config.load_balancing_strategy = None if strategy == LoadBalancingStrategy.FILL else strategy

    import logging

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("main", log_file=log_file, history_log_file=history_log_file, level=log_level)
    logger.info("Session log: %s", log_file)
    logger.info("History log: %s", history_log_file)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    npu_config_file = None
    if args.npu_config:
        config_path = Path(args.npu_config)
        candidates = []
        try:
            candidates.append(config_path.resolve(strict=False))
        except Exception:
            candidates.append(config_path.absolute())
        project_root = NPU_ROOT.parent.parent
        candidates.append(project_root / args.npu_config)
        candidates.append(NPU_ROOT / "configs" / "npu_resources" / Path(args.npu_config).name)
        candidates.append(NPU_ROOT.parent / "configs" / "npu_resources" / Path(args.npu_config).name)

        for candidate in candidates:
            if candidate.exists():
                npu_config_file = str(candidate)
                logger.info("Found NPU config: %s", npu_config_file)
                break

        if npu_config_file is None:
            logger.error("NPU config file not found: %s", args.npu_config)
            for c in candidates:
                logger.error("  - %s", c)

    server = NPUServer(
        npu_config_file=npu_config_file,
        log_file_path=log_file,
        history_log_file_path=history_log_file,
    )

    if args.npus:
        for npu_id in args.npus:
            server.npu_manager.register_npu(
                npu_id=npu_id,
                mode=config.default_gpu_mode,
                memory_threshold=config.default_memory_threshold,
            )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user", exc_info=True)
    except Exception as e:
        logger.error("Server error: %s", e, exc_info=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
