import contextlib
import io
import sys
import os
import tempfile

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

class OutputCapture: # 这个project还没用上
    """用于捕获所有输出（包括子进程）的上下文管理器"""
    def __init__(self):
        self.captured_output = ""
        self.old_stdout = None
        self.old_stderr = None
        self.temp_file = None

    def __enter__(self):
        # 创建临时文件用于捕获输出
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.close()

        # 保存原始的文件描述符
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # 将stdout和stderr重定向到临时文件
        temp_fd = os.open(self.temp_file.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(temp_fd, 1)  # stdout
        os.dup2(temp_fd, 2)  # stderr
        os.close(temp_fd)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 刷新缓冲区
        sys.stdout.flush()
        sys.stderr.flush()

        # 恢复原始的文件描述符
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stdout)
        os.close(self.old_stderr)

        # 读取捕获的输出，使用错误处理来处理二进制数据
        try:
            with open(self.temp_file.name, 'r', encoding='utf-8') as f:
                self.captured_output = f.read()
        except UnicodeDecodeError:
            # 如果有二进制数据，使用错误替换模式
            with open(self.temp_file.name, 'r', encoding='utf-8', errors='replace') as f:
                self.captured_output = f.read()

        # 清理临时文件
        os.unlink(self.temp_file.name)

    def get_output(self):
        return self.captured_output

def get_dir_subdirs(dir_path: str) -> list[str]:
    import os
    subdirs = []
    for d in os.listdir(dir_path): # os.listdir -> list[str]
        if os.path.isdir(os.path.join(dir_path, d)):
            subdirs.append(d)
    # return {f.name: f.path for f in os.scandir(dir_path) if f.is_dir()} # f: os.DirEntry(name, path)
    return subdirs

def get_dir_nums(dir_path: str) -> int:
    subdirs = get_dir_subdirs(dir_path)
    return len(subdirs)

# if __name__ == "__main__":
#     with capture_output() as (stdout_capture, stderr_capture):
#         print("Hello, world!")
#         print("Hello, world!", file=sys.stderr)
#     print(stdout_capture.getvalue())
#     print(stderr_capture.getvalue())

if __name__ == "__main__":
    from pprint import pprint
    subdirs = get_dir_subdirs("/workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op")
    print(get_dir_nums("/workspace/cu2tri/outputs/cu2tri/kernelbench_c/02_fused_op"))
    # pprint(subdirs)
    # for subdir in subdirs:
    #     print(subdir)
    #     print(subdirs[subdir])