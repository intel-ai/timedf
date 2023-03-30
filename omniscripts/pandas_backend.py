"""Holder of pandas backend. Intended use:
    1. Set correct pandas backend with `set_backend` call **before** any benchmark import.
    2. Get pandas in each benchmark module with `from utils.pandas_backend import pd`, this will use
     correct version of backend.
"""
# This will be replaced by modin.pandas after set_backend call
import pandas as pd  # noqa: F401 this import exists to provide vscode support for backend users

from env_manager.arg_parser import supported_backends
from .modin_utils import (
    import_pandas_into_module_namespace,
    trigger_execution_base as _trigger_execution_pandas,
)

__all__ = ["Backend"]


pandas_backends = [
    "Pandas",
    "Modin_on_ray",
    "Modin_on_dask",
    "Modin_on_python",
    "Modin_on_hdk",
]

nonpandas_backends = [
    "polars",
]

if sorted(supported_backends) != sorted([*pandas_backends, *nonpandas_backends]):
    raise ValueError(
        "Discovered inconsistency in supported backends\n"
        f"According to argparser supported backends are: {sorted(supported_backends)}\n"
        "According to backend module supported backends are: "
        f"{sorted([*pandas_backends, nonpandas_backends])}"
    )


class Backend:
    """Singleton storing backend utilities and configurations"""

    _supported_backends = supported_backends

    # Backend was initalized and ready for work
    _ready = False
    # Backend name
    _name = None

    # Modin config, none if pandas is used
    # Variable will hold the state, used for `trigger_execution`
    _modin_cfg = None

    @staticmethod
    def init(backend_name: str, ray_tmpdir=None, ray_memory=None):
        Backend._name = backend_name

        if backend_name in pandas_backends and backend_name != "Pandas":
            import modin.config as cfg

            Backend._modin_cfg = cfg

        if backend_name == "polars":
            pass
        elif backend_name == "Pandas":
            pass
        elif backend_name in pandas_backends:
            import_pandas_into_module_namespace(
                namespace=globals(),
                mode=backend_name,
                ray_tmpdir=ray_tmpdir,
                ray_memory=ray_memory,
            )
        else:
            raise ValueError(f"Unrecognized backend: {backend_name}")

        Backend._ready = True

    @staticmethod
    def _check_ready():
        if not Backend._ready:
            raise ValueError("Attempting to use unitialized backend")

    @staticmethod
    def get_name():
        Backend._check_ready()
        return Backend._name

    @staticmethod
    def get_modin_cfg():
        Backend._check_ready()
        return Backend._modin_cfg

    @staticmethod
    def trigger_execution(*dfs):
        """Utility function to trigger execution for lazy pd libraries. Returns actualized dfs."""
        Backend._check_ready()

        if Backend.get_name() == "polars":
            # Collect lazy frames
            results = [d.collect() if hasattr(d, 'collect') else d for d in dfs]
        elif Backend.get_name() in pandas_backends:
            _trigger_execution_pandas(*dfs, modin_cfg=Backend.get_modin_cfg())
            results = [*dfs]
        else:
            raise ValueError(f"no implementation for {Backend.get_name()}")

        if len(dfs) == 1:
            return results[0]
        else:
            return results
