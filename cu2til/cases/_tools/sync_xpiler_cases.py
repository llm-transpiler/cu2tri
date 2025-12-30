from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KERNEL_STUB = """
extern "C" void cuda_kernel(float *A, float *B, float *C, int size) {
  dim3 blockSize(960);
  dim3 numBlocks((size + 960 - 1) / 960);
  _cuda_kernel_impl<<<numBlocks, blockSize>>>(A, B, C);
}
""".strip()


@dataclass
class SyncOptions:
    dry_run: bool = False
    run_check: bool = False
    verbose: bool = False


def project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4]


def iter_case_dirs() -> Iterable[Path]:
    root = project_root() / "cu2til" / "cases" / "xpiler"
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.name.startswith("_"):
            yield path


def ensure_dir(path: Path, *, opts: SyncOptions) -> None:
    if opts.verbose:
        print(f"[ensure_dir] {path}")
    if opts.dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str, *, opts: SyncOptions) -> None:
    if opts.verbose:
        print(f"[write] {path}")
    if opts.dry_run:
        return
    path.write_text(content, encoding="utf-8")


def copy_file(src: Path, dst: Path, *, opts: SyncOptions) -> None:
    if opts.verbose:
        print(f"[copy] {src} -> {dst}")
    if opts.dry_run:
        return
    ensure_dir(dst.parent, opts=opts)
    shutil.copy2(src, dst)


def generate_kernel_from_ref(ref_src: Path, kernel_dst: Path, *, opts: SyncOptions) -> None:
    # Minimal heuristic: if ref contains a __global__ kernel named add, wrap as _cuda_kernel_impl
    content = ref_src.read_text(encoding="utf-8")
    # Replace kernel name to _cuda_kernel_impl if present
    content = re.sub(r"__global__\s+void\s+__launch_bounds__\([^)]*\)\s*(\w+)\(",
                     "__global__ void __launch_bounds__(960)\n    _cuda_kernel_impl(", content)
    # Append a simple host launcher (block/thread config) like the example
    ensure_dir(kernel_dst.parent, opts=opts)
    write_text(kernel_dst, content.strip() + "\n\n" + KERNEL_STUB + "\n", opts=opts)


def ensure_check_scripts(case_dir: Path, *, opts: SyncOptions) -> None:
    tools_root = project_root() / "cu2til" / "tools"
    src_cuda = tools_root / "check_cuda.py"
    src_triton = tools_root / "check_triton.py"

    dst_cuda = case_dir / "check_cuda.py"
    if not dst_cuda.exists():
        copy_file(src_cuda, dst_cuda, opts=opts)
    dst_triton = case_dir / "check_triton.py"
    if not dst_triton.exists() and src_triton.exists():
        copy_file(src_triton, dst_triton, opts=opts)


def ensure_torch_ref(case_dir: Path, op: str, *, opts: SyncOptions) -> None:
    torch_dir = case_dir / "torch_"
    ref_py = torch_dir / "ref.py"
    if ref_py.exists():
        return
    ensure_dir(torch_dir, opts=opts)

    # Basic op mapping; many cases can use idiomatic PyTorch kernels
    body = None
    if op == "add":
        body = "import torch\n\n\n" \
               "def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n" \
               "    return a + b\n"
    elif op in {"relu", "sigmoid", "gelu", "softmax", "sign"}:
        body = (
            "import torch\n\n\n"
            "def torch_kernel(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:\n"
            f"    return torch.{op}(x)\n"
        )
    elif op in {"sum"}:
        body = (
            "import torch\n\n\n"
            "def torch_kernel(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:\n"
            "    return torch.sum(x, dim=dim)\n"
        )
    else:
        # Fallback: identity placeholder; many ops will require manual impl tied to get_data.py
        body = (
            "import torch\n\n\n"
            "def torch_kernel(*args, **kwargs):\n"
            "    raise NotImplementedError('Please implement torch reference for this op')\n"
        )

    write_text(ref_py, body, opts=opts)


def process_case(case_dir: Path, *, opts: SyncOptions) -> None:
    name = case_dir.name
    op = name.split("_", 1)[0]

    cuda_dir = case_dir / "cuda_"
    ref_cu = cuda_dir / "ref.cu"
    kernel_cu = cuda_dir / "kernel.cu"

    if ref_cu.exists() and not kernel_cu.exists():
        generate_kernel_from_ref(ref_cu, kernel_cu, opts=opts)

    ensure_check_scripts(case_dir, opts=opts)
    ensure_torch_ref(case_dir, op, opts=opts)

    if opts.run_check and not opts.dry_run:
        try:
            subprocess.check_call([sys.executable, "check_cuda.py", "--no-perf"], cwd=str(case_dir))
            if opts.verbose:
                print(f"[ok] {name}")
        except subprocess.CalledProcessError:
            print(f"[FAIL] {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sync xpiler cases: generate missing files and validate")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    parser.add_argument("--run-check", action="store_true", help="Run check_cuda.py --no-perf for each case")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = parser.parse_args(argv)

    opts = SyncOptions(dry_run=args.dry_run, run_check=args.run_check, verbose=args.verbose)
    for case_dir in iter_case_dirs():
        process_case(case_dir, opts=opts)


if __name__ == "__main__":
    main()


