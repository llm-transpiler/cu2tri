"""Unittests for NVGPU examples.

Each test imports the corresponding example module and executes its main
function (or exported test function), asserting a zero return code.

Tests are skipped automatically if the NVGPU server is not reachable.
"""

