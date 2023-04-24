"""Holder of pandas backend. Intended use:
    1. Set correct pandas backend with `set_backend` call **before** any benchmark import.
    2. Get pandas in each benchmark module with `from utils.pandas_backend import pd`, this will use
     correct version of backend.
"""
from pathlib import Path

# This will be replaced by modin.pandas after set_backend call
import pandas as pd  # noqa: F401 this import exists to provide vscode support for backend users

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

supported_backends = pandas_backends + nonpandas_backends


class Backend:
    """Singleton storing backend utilities and configurations"""

    supported_backends = supported_backends

    # Backend was initalized and ready for work
    _ready = False
    # Backend name
    _name = None

    # Modin config, none if pandas is used
    # Variable will hold the state, used for `trigger_execution`
    _modin_cfg = None

    @classmethod
    def init(cls, backend_name: str, ray_tmpdir=None, ray_memory=None):
        cls._name = backend_name

        if backend_name in pandas_backends and backend_name != "Pandas":
            import modin.config as cfg

            cls._modin_cfg = cfg

        if backend_name == "polars":
            pass
        elif backend_name == "Pandas":
            pass
        elif backend_name in pandas_backends:
            Path(ray_tmpdir).mkdir(parents=True, exist_ok=True)
            import_pandas_into_module_namespace(
                namespace=globals(),
                mode=backend_name,
                ray_tmpdir=ray_tmpdir,
                ray_memory=ray_memory,
            )
        else:
            raise ValueError(f"Unrecognized backend: {backend_name}")

        cls._ready = True

    @classmethod
    def _check_ready(cls):
        if not cls._ready:
            raise ValueError("Attempting to use unitialized backend")

    @classmethod
    def get_name(cls):
        cls._check_ready()
        return cls._name

    @classmethod
    def get_modin_cfg(cls):
        cls._check_ready()
        return cls._modin_cfg

    @classmethod
    def trigger_execution(cls, *dfs):
        """Utility function to trigger execution for lazy pd libraries. Returns actualized dfs."""
        cls._check_ready()

        if cls.get_name() == "polars":
            # Collect lazy frames
            results = [d.collect() if hasattr(d, "collect") else d for d in dfs]
        elif cls.get_name() in pandas_backends:
            _trigger_execution_pandas(*dfs, modin_cfg=cls.get_modin_cfg())
            results = [*dfs]
        else:
            raise ValueError(f"no implementation for {cls.get_name()}")

        if len(dfs) == 1:
            return results[0]
        else:
            return results
