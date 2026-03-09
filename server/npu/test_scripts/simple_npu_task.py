import os
import json

payload = {
    "ASCEND_VISIBLE_DEVICES": os.getenv("ASCEND_VISIBLE_DEVICES"),
    "ASCEND_RT_VISIBLE_DEVICES": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
}
print(json.dumps(payload, ensure_ascii=True))
