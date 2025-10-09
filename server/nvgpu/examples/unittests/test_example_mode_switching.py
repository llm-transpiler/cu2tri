import time
import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleModeSwitching(unittest.TestCase):
    def test_shared_to_exclusive_waits_new_tasks(self):
        client = get_client()
        gpus = client.list_gpus()
        self.assertGreater(len(gpus), 0)
        gpu_id = gpus[0]["gpu_id"]

        # Ensure shared and start two tasks
        client.set_gpu_mode(gpu_id, "shared", manual=True)
        ids = []
        for _ in range(2):
            tid = client.submit_task(
                script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
                task_type="functional",
                args=["--duration", "10"],
                gpu_id=gpu_id,
            )
            ids.append(tid)
            time.sleep(0.2)

        # Wait until at least one running
        for _ in range(20):
            g = client.get_gpu(gpu_id)
            if g["running_task_count"] >= 1:
                break
            time.sleep(0.5)

        # Switch to exclusive; submit new task which should wait until others finish
        client.set_gpu_mode(gpu_id, "exclusive", manual=True)
        new_task = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional",
            gpu_id=gpu_id,
        )
        time.sleep(0.5)
        status_new = client.get_task(new_task).status
        self.assertIn(status_new, ("pending", "queued"))

        # Wait for previous tasks to finish, then new task should complete
        for tid in ids:
            try:
                client.wait_for_task(tid, timeout=60, poll_interval=0.5)
            except Exception:
                pass
        final_new = client.wait_for_task(new_task, timeout=120, poll_interval=0.5)
        self.assertEqual(final_new.status, "completed")

        # Restore shared
        client.set_gpu_mode(gpu_id, "shared", manual=True)
