import unittest
import time
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleErrorHandling(unittest.TestCase):
    def test_failing_exit_code(self):
        client = get_client()
        tid = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/failing_test.py",
            task_type="functional",
            args=["--error-code", "42", "--error-message", "UnitTestFail"],
        )
        final = client.wait_for_task(tid, timeout=60, poll_interval=0.5)
        self.assertEqual(final.status, "failed")
        self.assertEqual(final.exit_code, 42)
        # stderr will contain the message; through API we can only see summary; ensure log exists
        self.assertIsNotNone(final.log_file)

    def test_timeout_and_force_cancel(self):
        client = get_client()
        tid = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "60"],
        )
        # Expect timeout on short wait
        with self.assertRaises(TimeoutError):
            client.wait_for_task(tid, timeout=2, poll_interval=0.5)
        # Then force cancel and ensure cancelled
        client.cancel_task(tid, force=True)
        time.sleep(0.5)
        res = client.get_task(tid)
        self.assertEqual(res.status, "cancelled")
