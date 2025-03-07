import importlib
import importlib.util
from types import ModuleType

from rl_the_spire.utils.add_to_sys_path_context import add_to_sys_path


def import_module_with_syspath(module_name: str, module_path: str) -> ModuleType:
    """
    Imports a module from the specified path without polluting the global sys.path.

    Parameters:
    - module_name: Name to assign to the imported module.
    - module_path: Path to the directory containing the module.

    Returns:
    - The imported module object.
    """
    with add_to_sys_path(module_path):
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ImportError(
                f"Cannot find module named '{module_name}' in '{module_path}'"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module
