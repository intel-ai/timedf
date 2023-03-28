"""Holder of pandas backend. Intended use:
    1. Set correct pandas backend with `set_backend` call **before** any benchmark import.
    2. Get pandas in each benchmark module with `from utils.pandas_backend import pd`, this will use
     correct version of backend.
"""
import pandas as pd  # noqa: F401 this import exists to provide vscode support for backend users

from .modin_utils import (
    import_pandas_into_module_namespace,
    trigger_execution_base as _trigger_execution_pandas,
)

# Modin config, none if pandas is used
# Variable will hold the state, used for `trigger_execution`
modin_cfg = None
backend_cfg = {}


def _get_modin_config(pandas_mode):
    if pandas_mode != "Pandas":
        import modin.config as cfg

        return cfg
    else:
        return None


def set_backend(pandas_mode, ray_tmpdir, ray_memory):
    global modin_cfg
    global backend_cfg
    backend_cfg["backend"] = pandas_mode

    modin_cfg = _get_modin_config(pandas_mode)
    import_pandas_into_module_namespace(
        namespace=globals(), mode=pandas_mode, ray_tmpdir=ray_tmpdir, ray_memory=ray_memory
    )


def trigger_execution(*dfs):
    """Utility function to trigger execution for lazy pd libraries."""
    _trigger_execution_pandas(*dfs, modin_cfg=modin_cfg)
    if len(dfs) == 1:
        return dfs[0]
    else:
        return dfs
