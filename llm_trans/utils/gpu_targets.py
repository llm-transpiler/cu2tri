from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GPUTarget:
    key: str
    label: str
    architecture: str
    compute: str
    sm: str
    cutlass_arch: str
    memory: str
    notes: List[str]
    default_gpu_index: int | None = None

    @property
    def nvcc_gencode(self) -> str:
        return f"arch={self.compute},code={self.sm}"

    def summary_line(self) -> str:
        return f"{self.label} ({self.architecture}, {self.memory})"


GPU_TARGETS: Dict[str, GPUTarget] = {
    "h800_sxm": GPUTarget(
        key="h800_sxm",
        label="NVIDIA H800 80GB SXM5",
        architecture="Hopper SM90",
        compute="compute_90",
        sm="sm_90",
        cutlass_arch="SM90",
        memory="80GB HBM3 @ 2.3TB/s",
        notes=[
            "Prefer 128-thread tiled mainloop with Tensor Cores (MMA1688)",
            "Use TMA async copies and double buffering for global->shared moves",
            "Shared memory up to 228KB per CTA; schedule for high occupancy",
        ],
        default_gpu_index=7,
    ),
    "h800_pcie": GPUTarget(
        key="h800_pcie",
        label="NVIDIA H800 80GB PCIe",
        architecture="Hopper SM90",
        compute="compute_90",
        sm="sm_90",
        cutlass_arch="SM90",
        memory="80GB HBM3 @ 1.7TB/s",
        notes=[
            "Lower NVLink bandwidth than SXM; avoid unnecessary host syncs",
            "Keep per-CTA shared memory under 164KB for better residency",
        ],
    ),
    "h100_pcie": GPUTarget(
        key="h100_pcie",
        label="NVIDIA H100 80GB PCIe",
        architecture="Hopper SM90",
        compute="compute_90",
        sm="sm_90",
        cutlass_arch="SM90",
        memory="80GB HBM3 @ 2.0TB/s",
        notes=[
            "Favour Tensor Core MMA and TMA pipelines",
            "Balance register usage to keep ≥2 CTAs/SM",
        ],
    ),
    "rtx6000_ada": GPUTarget(
        key="rtx6000_ada",
        label="NVIDIA RTX 6000 Ada",
        architecture="Ada SM89",
        compute="compute_89",
        sm="sm_89",
        cutlass_arch="SM89",
        memory="48GB GDDR6 @ 960GB/s",
        notes=[
            "SM89 lacks TMA; rely on cp.async bulk and Tensor Core MMA168p",
            "Prefer contiguous global accesses to saturate GDDR bandwidth",
        ],
    ),
    "a800_sxm": GPUTarget(
        key="a800_sxm",
        label="NVIDIA A800 80GB SXM",
        architecture="Ampere SM80",
        compute="compute_80",
        sm="sm_80",
        cutlass_arch="SM80",
        memory="80GB HBM2e @ 2.0TB/s",
        notes=[
            "Tensor Core MMA884 (TF32/BF16) provide best throughput",
            "Avoid excessive shared memory to keep ≥2 CTAs active",
        ],
    ),
    "rtx5090": GPUTarget(
        key="rtx5090",
        label="NVIDIA RTX 5090",
        architecture="Blackwell SM92",
        compute="compute_92",
        sm="sm_92",
        cutlass_arch="SM90",  # CUTLASS 3.5 treats Blackwell similar to SM90
        memory="32GB GDDR7 (projected >1.5TB/s)",
        notes=[
            "Assume next-gen Tensor Cores (MMA16912) with large register file",
            "Plan for high clock variance; keep kernels latency tolerant",
        ],
    ),
}


def get_gpu_target(key: str) -> GPUTarget:
    return GPU_TARGETS.get(key, GPU_TARGETS["h800_sxm"])


def available_gpu_targets() -> List[str]:
    return sorted(GPU_TARGETS.keys())


def format_prompt_hint(target: GPUTarget) -> str:
    bullets = "\n".join(f"- {note}" for note in target.notes)
    return (
        f"Target GPU: {target.label} ({target.architecture}, {target.memory}).\n"
        f"Key optimization reminders:\n{bullets}"
    )


__all__ = ["GPUTarget", "GPU_TARGETS", "get_gpu_target", "available_gpu_targets", "format_prompt_hint"]
