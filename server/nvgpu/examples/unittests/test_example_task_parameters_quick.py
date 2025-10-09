import unittest
from server.nvgpu.examples.unittests.utils import requires_server


@requires_server
class TestExampleTaskParametersQuick(unittest.TestCase):
    def test_task_parameters_quick(self):
        from server.nvgpu.examples import example_task_parameters_quick as mod
        rc = mod.main()
        self.assertEqual(rc, 0)
