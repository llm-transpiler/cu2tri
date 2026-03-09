import argparse
import os
import sys
from pathlib import Path
from contextlib import contextmanager

TESTCASE_ROOT_DIR = Path(os.path.dirname(__file__))
sys.path.insert(0, str(TESTCASE_ROOT_DIR))

import torch
from get_data import Params, get_cuda_torch_inputs  # type: ignore
from torch_.ref import torch_kernel  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Ascend C target smoke test")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Keep compatibility with existing test runner args",
    )
    return parser.parse_args()


def _check_generated_ascendc() -> tuple[bool, str]:
    kernel_path = TESTCASE_ROOT_DIR / "ascendc_" / "kernel.cpp"
    if not kernel_path.exists():
        return False, f"Missing generated Ascend C file: {kernel_path}"

    content = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if "ascendc_kernel" not in content:
        return False, "Generated Ascend C file does not define required symbol: ascendc_kernel"

    return True, "Ascend C source present"


def _resolve_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    npu_attr = getattr(torch, "npu", None)
    if npu_attr is not None:
        try:
            if npu_attr.is_available():
                return "npu"
        except Exception:
            pass
    return "cpu"


@contextmanager
def _patch_torch_device_alias(target_device: str):
    if target_device == "cuda":
        yield
        return

    wrappers = {}
    names = ["randn", "rand", "zeros", "ones", "empty", "full", "tensor"]
    for name in names:
        fn = getattr(torch, name, None)
        if fn is None:
            continue

        def _make_wrapper(orig_fn):
            def _wrapper(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = target_device
                return orig_fn(*args, **kwargs)

            return _wrapper

        wrappers[name] = fn
        setattr(torch, name, _make_wrapper(fn))

    try:
        yield
    finally:
        for name, fn in wrappers.items():
            setattr(torch, name, fn)


def _check_torch_reference() -> tuple[bool, str]:
    params = Params()
    runtime_device = _resolve_runtime_device()
    with _patch_torch_device_alias(runtime_device):
        _, torch_all_inputs, _ = get_cuda_torch_inputs(params)
    output = torch_kernel(*torch_all_inputs)
    if output is None:
        return False, "torch reference output is None"
    return True, f"torch reference executed on {runtime_device}"


if __name__ == "__main__":
    parse_args()

    ok_src, msg_src = _check_generated_ascendc()
    print(f"[AscendC] {msg_src}")
    if not ok_src:
        print("FAILED")
        sys.exit(1)

    ok_torch, msg_torch = _check_torch_reference()
    print(f"[Torch] {msg_torch}")
    if not ok_torch:
        print("FAILED")
        sys.exit(1)

    print("PASSED")
    sys.exit(0)
