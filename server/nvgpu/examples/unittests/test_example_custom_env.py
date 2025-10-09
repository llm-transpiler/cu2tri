import unittest
from server.nvgpu.examples.unittests.utils import requires_server, get_client


@requires_server
class TestExampleCustomEnv(unittest.TestCase):
    def test_env_propagation_to_task(self):
        client = get_client()

        # Create a simple test script that prints selected envs
        test_script = "/tmp/env_test_unittest.py"
        with open(test_script, "w", encoding="utf-8") as f:
            f.write(
                "#!/usr/bin/env python3\n"
                "import os; print('MY_VAR=', os.environ.get('MY_VAR', ''))\n"
                "print('TEST_MODE=', os.environ.get('TEST_MODE', ''))\n"
            )

        env = {"MY_VAR": "hello", "TEST_MODE": "unittest"}
        tid = client.submit_task(script_path=test_script, task_type="functional", env=env)
        final = client.wait_for_task(tid, timeout=60, poll_interval=0.5)
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.exit_code, 0)

        stdout = client.get_task_log(tid, log_type="stdout", limit=4096)
        content = stdout.get("content", "")
        self.assertIn("MY_VAR= hello", content)
        self.assertIn("TEST_MODE= unittest", content)
