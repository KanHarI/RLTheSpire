import os
import sys
from contextlib import contextmanager
from typing import Generator


@contextmanager
def add_to_sys_path(path: str) -> Generator[None, None, None]:
    """
    Context manager to temporarily add a directory to sys.path.
    """
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)
        added = True
    else:
        added = False
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)
