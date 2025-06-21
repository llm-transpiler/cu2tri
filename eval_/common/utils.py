import os


def cleanup_temp_file(file_path: str):
    try:
        os.unlink(file_path)
    except:
        pass
