import contextlib
import io
import sys

@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr output"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    
    try:
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

if __name__ == "__main__":
    with capture_output() as (stdout_capture, stderr_capture):
        print("Hello, world!")
        print("Hello, world!", file=sys.stderr)
    print(stdout_capture.getvalue())
    print(stderr_capture.getvalue())