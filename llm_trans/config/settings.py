from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import dotenv
import os

from server.common.timezone import apply_default_timezone_to_os, format_timestamp


from .model_registry import ModelRegistry, ModelRegistryError, load_model_registry
from ..utils.gpu_targets import GPUTarget, get_gpu_target


@dataclass
class Settings:
    args: argparse.Namespace
    temperature: float = 0.35
    max_rounds: int = 5
    timestamp: str = field(default_factory=format_timestamp)
    base_output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    outputs_root: Path | None = None
    dir_cuda: Path = Path("cuda_")
    dir_torch: Path = Path("torch_")
    dir_triton: Path = Path("triton_")
    dir_cute: Path = Path("cute_")
    direction: str = field(init=False)
    testset_root_dir: Path = field(init=False)
    all_cases: Dict[str, list[str]] = field(init=False)
    check_suffix: str = field(init=False)
    console_output: bool = field(init=False)
    use_nvgpu: bool = field(init=False)
    resume_conversation: bool = field(init=False)
    model_name: str | None = None
    model_name_clean: str | None = None
    work_dir: Path | None = None
    log_file: Path | None = None
    jsonl_file: Path | None = None
    model_registry: ModelRegistry = field(init=False)
    model_registry_path: Path = field(init=False)
    project_root: Path = field(init=False)
    target_gpu: str = field(init=False)
    gpu_target: GPUTarget = field(init=False)
    test_gpu_index: int | None = field(init=False)

    def __post_init__(self) -> None:
        if self.args.max_attempts < 1:
            raise ValueError("--max-attempts must be at least 1")
        max_rounds = getattr(self.args, "max_rounds", self.max_rounds)
        if max_rounds < 1:
            raise ValueError("--max-rounds must be at least 1")
        self.max_rounds = max_rounds

        self.console_output = bool(self.args.console and not self.args.no_console)
        self.use_nvgpu = bool(self.args.use_nvgpu and not self.args.no_nvgpu)
        self.resume_conversation = bool(getattr(self.args, "resume_conversation", False))
        
        # Translation direction
        self.direction = getattr(self.args, "direction", None)
        if self.direction is None:
            raise ValueError("Transpile direction is not set. Please use --direction to set the translation direction.")

        project_root_env = os.getenv("PROJECT_ROOT")
        if project_root_env:
            self.project_root = Path(project_root_env)
        else:  # pragma: no cover - fallback for tests
            self.project_root = Path(__file__).resolve().parents[3]

        target_gpu_key = getattr(self.args, "target_gpu", "h800_sxm")
        self.gpu_target = get_gpu_target(target_gpu_key)
        self.target_gpu = self.gpu_target.key
        cli_gpu_index = getattr(self.args, "test_gpu", None)
        if cli_gpu_index is not None:
            self.test_gpu_index = cli_gpu_index
        else:
            self.test_gpu_index = self.gpu_target.default_gpu_index

        self.testset_root_dir = self.project_root / Path(f"llm_trans/cases/{self.args.testset}")

        self.check_suffix = "" if "xpiler" in self.args.testset else "_dynamic"
        from ..services.cases import resolve_cases_for_testset
        self.all_cases = resolve_cases_for_testset(self.args.testset)

        model_config_path = os.getenv("LLM_MODEL_CONFIG")
        if model_config_path:
            self.model_registry_path = Path(model_config_path)
        else:
            self.model_registry_path = self.project_root / Path("llm_trans/config/model_clients.yaml")

        try:
            self.model_registry = load_model_registry(self.model_registry_path)
        except ModelRegistryError as exc:  # pragma: no cover - configuration error
            raise RuntimeError(str(exc)) from exc

        # Resolve outputs_root with priority: CLI > env > default
        # Use direction to determine output subdirectory
        cli_root = getattr(self.args, "outputs_root", None)
        if cli_root:
            self.outputs_root = Path(cli_root).expanduser().resolve()
        else:
            self.outputs_root = self.project_root / Path(f"llm_trans/runs/{self.direction}")

        # Backward-compat alias
        self.base_output_dir = self.outputs_root

    def configure_output_paths(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_name_clean = "".join(
            c if c.isalnum() else "_"
            for c in model_name.split("/")[-1].lower()
        )
        # work_dir structure: {outputs_root}/{testset}/{model_name}/{timestamp}
        if self.outputs_root is None:
            raise RuntimeError("outputs_root is not configured")
        self.work_dir = self.outputs_root / self.args.testset / self.model_name_clean / self.timestamp
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.work_dir / f"{self.model_name_clean}_{self.args.testset}.log"
        self.jsonl_file = self.work_dir / f"{self.model_name_clean}_{self.args.testset}.jsonl"


def build_settings(argv: list[str] | None = None) -> Settings:
    dotenv.load_dotenv()
    apply_default_timezone_to_os()
    from .args import parse_cli_args

    args = parse_cli_args(argv)
    return Settings(args=args)


__all__ = ["Settings", "build_settings"]
