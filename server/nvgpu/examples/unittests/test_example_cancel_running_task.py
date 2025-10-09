import re
import time
import unittest
from server.nvgpu.examples.unittests.utils import (
    requires_server,
    get_client,
    get_server_log_path,
    get_file_size,
    read_new_log_content,
)


@requires_server
class TestExampleCancelRunningTask(unittest.TestCase):
    def test_force_cancel_running_task(self):
        client = get_client()

        server_log = get_server_log_path()
        pre = get_file_size(server_log) if server_log else 0

        # Submit a long running task
        task_id = client.submit_task(
            script_path="/workspace/server/nvgpu/test_scripts/long_running_task.py",
            task_type="functional",
            args=["--duration", "60", "--report-interval", "5"],
        )

        # Wait until running
        for _ in range(40):
            r = client.get_task(task_id)
            if r.status == "running":
                break
            time.sleep(0.5)
        self.assertIn(r.status, ("running", "queued", "pending"))

        # Force cancel
        client.cancel_task(task_id, force=True)

        # Verify cancelled
        time.sleep(1.0)
        final = client.get_task(task_id)
        self.assertEqual(final.status, "cancelled")
        self.assertIsNotNone(final.error_message)
        self.assertIn("Cancelled", final.error_message)

        # Server log must mention force cancelled for this task
        if server_log and server_log.exists():
            tail = read_new_log_content(server_log, pre)
            self.assertRegex(
                tail,
                re.compile(rf"Task {re.escape(task_id)} .*force cancelled", re.I | re.S),
            )
