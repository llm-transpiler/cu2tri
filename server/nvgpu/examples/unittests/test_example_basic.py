import os
import re
import time
import unittest
from pathlib import Path

from server.nvgpu.examples.unittests.utils import (
    requires_server,
    get_client,
    get_server_log_path,
    get_file_size,
    read_new_log_content,
)


@requires_server
class TestExampleBasic(unittest.TestCase):
    def test_basic_functional_task_end_to_end(self):
        """Submit a simple functional task and validate results + server logs."""
        client = get_client()

        # Track server log tail for this test
        server_log = get_server_log_path()
        pre_size = get_file_size(server_log) if server_log else 0

        # Submit task with explicit type to assert smart default mode
        script = "/workspace/server/nvgpu/test_scripts/simple_functional_test.py"
        task_id = client.submit_task(script_path=script, task_type="functional")

        # Wait for completion
        result = client.wait_for_task(task_id, timeout=120, poll_interval=0.5)

        # Basic result assertions
        self.assertEqual(result.task_id, task_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.exit_code, 0)
        # Smart default: functional -> shared
        self.assertIn(result.task_mode, ("shared", None))  # some versions may omit

        # Log file assertions
        self.assertIsNotNone(result.log_file)
        log_path = Path(result.log_file)
        self.assertTrue(log_path.exists(), f"log not found: {log_path}")

        content = log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("=== Simple Functional Test ===", content)
        # At least one of the success markers must appear
        self.assertTrue(
            ("Matrix multiplication: PASSED" in content) or ("ALL TESTS PASSED" in content),
            "expected success markers not found in task log",
        )

        # Server log assertions (if available)
        if server_log and server_log.exists():
            time.sleep(0.2)
            new_tail = read_new_log_content(server_log, pre_size)
            # Submitted line includes id and script
            self.assertRegex(
                new_tail,
                re.compile(rf"TASK {re.escape(task_id)} submitted: .*simple_functional_test\.py", re.S),
            )
            # Completed line includes id and completed on GPU
            self.assertRegex(
                new_tail,
                re.compile(rf"TASK {re.escape(task_id)} completed on GPU", re.S),
            )
