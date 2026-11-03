import atexit
import shutil
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "telegram_modules"

telegram_modules = types.ModuleType("telegram_modules")
telegram_modules.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("telegram_modules", telegram_modules)

utils_package = types.ModuleType("telegram_modules.utils")
utils_package.__path__ = [str(PACKAGE_ROOT / "utils")]
sys.modules.setdefault("telegram_modules.utils", utils_package)

config_module = types.ModuleType("telegram_modules.config")
config_module.NODE_CACHE_DIR = tempfile.mkdtemp(prefix="telegram-sender-test-cache-")
atexit.register(shutil.rmtree, config_module.NODE_CACHE_DIR, ignore_errors=True)
sys.modules.setdefault("telegram_modules.config", config_module)

log_module = types.ModuleType("telegram_modules.utils.log")
log_module.print_info = lambda message: None
log_module.print_warning = lambda message: None
log_module.print_error = lambda message: None
sys.modules.setdefault("telegram_modules.utils.log", log_module)

folder_paths = types.ModuleType("folder_paths")
folder_paths.roots = {}
folder_paths.get_folder_paths = lambda folder_type: folder_paths.roots.get(folder_type, [])
sys.modules.setdefault("folder_paths", folder_paths)
