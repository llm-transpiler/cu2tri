"""Timer 基础示例。

运行本文件可以看到默认 reporter 打印的耗时信息。
"""

from __future__ import annotations

import time

from profiler.timer import create_timer


timer = create_timer()


def warm_up_gpu() -> None:
    """模拟 GPU 预热。"""
    time.sleep(0.01)


def export_model(path: str) -> str:
    """模拟模型导出。"""
    time.sleep(0.02)
    return path


@timer.wrap("batch_inference")
def run_batch(batch_id: int) -> str:
    """使用装饰器测量函数耗时。"""
    time.sleep(0.005)
    return f"batch-{batch_id}"


def main() -> None:
    with timer.time("warmup"):
        warm_up_gpu()

    batch = run_batch(1)
    print("batch result:", batch)

    model_path = timer.measure("export", export_model, path="model.onnx")
    print("model exported:", model_path)


if __name__ == "__main__":
    main()

'''
[timer] warmup: 10.058 ms (OK)
[timer] batch_inference: 5.056 ms (OK)
batch result: batch-1
[timer] export: 20.058 ms (OK)
model exported: model.onnx
'''