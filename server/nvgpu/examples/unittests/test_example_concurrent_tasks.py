import time
import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleConcurrentTasks(unittest.TestCase):
    def test_shared_concurrency_limit_respected(self):
        client = get_client()

        gpus = client.list_gpus()
        self.assertGreater(len(gpus), 0, "no GPUs registered")
        gpu_id = gpus[0]["gpu_id"]

        # Set shared mode and limit to 2
        self.assertTrue(client.set_gpu_mode(gpu_id, "shared", manual=True))
        self.assertTrue(client.set_gpu_max_concurrent_tasks(gpu_id, 2))

        # Submit 3 tasks that run briefly
        ids = []
        for _ in range(3):
            tid = client.submit_task(
                script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
                task_type="functional",
                args=["--duration", "12", "--report-interval", "4"],
                gpu_id=gpu_id,
            )
            ids.append(tid)
            time.sleep(0.2)

        # Observe running counts; ensure never > 2
        max_running = 0
        for _ in range(15):
            time.sleep(1)
            g = client.get_gpu(gpu_id)
            max_running = max(max_running, g["running_task_count"])
        self.assertLessEqual(max_running, 2, f"observed running_task_count {max_running} > 2")

        # Cleanup wait
        for tid in ids:
            try:
                client.wait_for_task(tid, timeout=60, poll_interval=0.5)
            except Exception:
                pass
