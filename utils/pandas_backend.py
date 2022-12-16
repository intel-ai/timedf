"""Holder of pandas backend. Intended use:
    1. Set correct pandas backend with `set_backend` call **before** any benchmark import.
    2. Get pandas in each benchmark module with `import pd from utils.pandas_backend, this will use
     correct version of backend.
"""
import pandas as pd

from .namespace_utils import import_pandas_into_module_namespace


def set_backend(pandas_mode, ray_tmpdir, ray_memory):
    import_pandas_into_module_namespace(
        namespace=globals(), mode=pandas_mode, ray_tmpdir=ray_tmpdir, ray_memory=ray_memory
    )
