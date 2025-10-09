import time
import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleGPUManagement(unittest.TestCase):
    def test_exclusive_blocks_second_task(self):
        client = get_client()
        gpus = client.list_gpus()
        self.assertGreater(len(gpus), 0)
        gpu_id = gpus[0]["gpu_id"]

        # Set to exclusive
        self.assertTrue(client.set_gpu_mode(gpu_id, "exclusive", manual=True))
        time.sleep(0.2)

        # Submit first task
        t1 = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional",
            gpu_id=gpu_id,
        )
        time.sleep(0.5)
        s1 = client.get_task(t1).status
        self.assertIn(s1, ("running", "completed"))

        # Submit second task; should not be able to run concurrently
        t2 = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional",
            gpu_id=gpu_id,
        )
        time.sleep(0.5)
        s2 = client.get_task(t2).status
        self.assertIn(s2, ("pending", "queued"))

        # Wait and ensure t1 finishes, then t2 can run/finish
        client.wait_for_task(t1, timeout=120, poll_interval=0.5)
        final2 = client.wait_for_task(t2, timeout=120, poll_interval=0.5)
        self.assertEqual(final2.status, "completed")

        # Restore shared
        self.assertTrue(client.set_gpu_mode(gpu_id, "shared", manual=True))
