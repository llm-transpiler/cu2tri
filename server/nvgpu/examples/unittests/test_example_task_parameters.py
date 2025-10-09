import time
import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleTaskParameters(unittest.TestCase):
    def test_smart_defaults_and_overrides(self):
        client = get_client()

        script = "/workspace/server/nvgpu/test_scripts/simple_functional_test.py"

        cases = [
            ("functional→shared", {"task_type": "functional"}, "shared"),
            ("performance→exclusive", {"task_type": "performance"}, "exclusive"),
            ("both→exclusive", {"task_type": "both"}, "exclusive"),
            ("override functional+exclusive", {"task_type": "functional", "task_mode": "exclusive"}, "exclusive"),
            ("override performance+shared", {"task_type": "performance", "task_mode": "shared"}, "shared"),
        ]

        for name, kwargs, expected_mode in cases:
            with self.subTest(name=name):
                task_id = client.submit_task(script_path=script, **kwargs)
                # Give the server a moment to populate fields
                time.sleep(0.3)
                res = client.get_task(task_id)
                self.assertEqual(res.task_mode, expected_mode,
                                 f"expected mode {expected_mode} for {name}, got {res.task_mode}")
                # Clean by waiting to finish to avoid leaking tasks
                final = client.wait_for_task(task_id, timeout=90, poll_interval=0.5)
                self.assertEqual(final.status, "completed")
