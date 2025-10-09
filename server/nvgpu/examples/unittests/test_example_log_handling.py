import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleLogHandling(unittest.TestCase):
    def test_log_api_summary_stdout_stderr(self):
        client = get_client()

        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/simple_functional_test.py",
            task_type="functional",
        )
        final = client.wait_for_task(task_id, timeout=120, poll_interval=0.5)
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.exit_code, 0)

        # Summary log
        summary = client.get_task_log(task_id, log_type="summary", limit=2048)
        self.assertEqual(summary["task_id"], task_id)
        self.assertEqual(summary["log_type"], "summary")
        self.assertIn("total_size", summary)
        self.assertIn("content", summary)
        self.assertGreaterEqual(summary["total_size"], len(summary["content"].encode("utf-8")))

        # Full summary should contain test banner
        full = client.get_full_task_log(task_id, log_type="summary")
        self.assertIn("Simple Functional Test", full)

        # stdout / stderr metadata structure should be valid even if empty
        stdout = client.get_task_log(task_id, log_type="stdout", limit=1024)
        self.assertEqual(stdout["task_id"], task_id)
        self.assertEqual(stdout["log_type"], "stdout")
        self.assertIn("total_size", stdout)
        self.assertIn("content", stdout)

        stderr = client.get_task_log(task_id, log_type="stderr", limit=1024)
        self.assertEqual(stderr["task_id"], task_id)
        self.assertEqual(stderr["log_type"], "stderr")
        self.assertIn("total_size", stderr)
        self.assertIn("content", stderr)
