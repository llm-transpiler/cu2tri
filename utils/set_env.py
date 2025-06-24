import os
from pathlib import Path
import sys

import dotenv
dotenv.load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if PROJECT_ROOT:
    PROJECT_ROOT = Path(PROJECT_ROOT)
else:
    # 如果环境变量未设置，使用相对路径
    PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def set_env():
    """设置CUDA和PyTorch相关的环境变量"""
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5"
    os.environ['TORCH_CUDA_ARCH_LIST'] = "Ada"

# Fix libstdc++ compatibility issue

def fix_libstdc_path():
    """Fix libstdc++ path to use system version instead of conda's old version"""
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    system_lib_paths = ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu']
    
    new_paths = [path for path in system_lib_paths if os.path.exists(path)]
    if current_ld_path:
        new_ld_path = ':'.join(new_paths + [current_ld_path])
    else:
        new_ld_path = ':'.join(new_paths)
    
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    print(f"Fixed LD_LIBRARY_PATH for libstdc++ compatibility")
