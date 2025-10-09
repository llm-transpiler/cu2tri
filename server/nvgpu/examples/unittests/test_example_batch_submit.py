import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleBatchSubmit(unittest.TestCase):
    def test_batch_submit_success(self):
        client = get_client()
        tasks = [
            {"script": "/workspace/server/nvgpu/test_scripts/simple_functional_test.py", "task_type": "functional", "args": []},
            {"script": "/workspace/server/nvgpu/test_scripts/memory_stress_test.py", "task_type": "functional", "args": ["--size", "256", "--iterations", "1"]},
            {"script": "/workspace/server/nvgpu/test_scripts/performance_benchmark.py", "task_type": "performance", "args": ["--matmul-size", "512", "--matmul-iters", "1", "--skip-memory"]},
        ]

        ids = [
            client.submit_task(script_path=t["script"], task_type=t["task_type"], args=t["args"]) for t in tasks
        ]

        # Wait for all
        finals = [client.wait_for_task(tid, timeout=300, poll_interval=0.5) for tid in ids]
        for f in finals:
            self.assertEqual(f.status, "completed")
            self.assertEqual(f.exit_code, 0)
