from contextlib import contextmanager
import pynvml

@contextmanager
def _nvml():
    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()

@_nvml()
def func():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    ...
    
with _nvml():
    cnt = pynvml.nvmlDeviceGetCount()
    print(cnt)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(pynvml.nvmlDeviceGetName(handle))
    print(pynvml.nvmlDeviceGetUUID(handle))
    print(pynvml.nvmlDeviceGetMemoryInfo(handle))