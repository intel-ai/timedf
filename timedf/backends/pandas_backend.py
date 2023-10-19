"""Holder of pandas backend. Intended use:
    1. Set correct pandas backend by initing pandas backend at least once **before** any benchmark import.
    2. Get pandas in each benchmark module with `from backends.pandas_backend import pd`, this will use
     correct version of pandas from [pandas, modin-on-hdk, modin-on-ray].
"""
from pathlib import Path

# This will be replaced by modin.pandas after Backend.init() call
import pandas as pd  # noqa: F401 this import exists to provide vscode support for backend users

from .modin_utils import (
    import_pandas_into_module_namespace,
    execute as execute_pandas,
)
from .. import backend

pandas_backends = [
    "Pandas",
    "Modin_on_ray",
    "Modin_on_dask",
    "Modin_on_python",
    "Modin_on_hdk",
    "Modin_on_unidist_mpi",
]


class PandasBackend:
    """Singleton storing backend utilities and configurations"""

    def __init__(self, params) -> None:
        # We need modin_cfg for legacy reasons
        # Variable will hold the state, used for `trigger_execution`
        # params = {
        #     "backend_name": backend_name,
        #     "ray_tmpdir": ray_tmpdir,
        #     "ray_memory": ray_memory,
        #     # TODO: is this necessary? This is a general parameter
        #     "num_threads": num_threads,
        # }
        self.pandas_mode = params["pandas_mode"]
        ray_tmpdir = params["ray_tmpdir"]
        ray_memory = params["ray_memory"]
        num_threads = params["num_threads"]

        # Modin config, none if pandas is used
        # Variable will hold the state, used for `trigger_execution`
        if self.pandas_mode != "Pandas":
            self._modin_cfg = self.params = params
        else:
            self._modin_cfg = self.params = None

        if self.pandas_mode not in pandas_backends:
            raise ValueError(f"Unrecognized pandas_mode: {self.pandas_mode}")

        if self.pandas_mode != "Pandas":
            Path(ray_tmpdir).mkdir(parents=True, exist_ok=True)
            import_pandas_into_module_namespace(
                namespace=globals(),
                mode=self.pandas_mode,
                ray_tmpdir=ray_tmpdir,
                ray_memory=ray_memory,
                num_threads=num_threads,
            )
            # # TODO: legacy for backward-compatibility
            # import_pandas_into_module_namespace(
            #     namespace=backend,
            #     mode=self.pandas_mode,
            #     ray_tmpdir=ray_tmpdir,
            #     ray_memory=ray_memory,
            #     num_threads=num_threads,
            # )
            backend.pd = pd

    def _trigger_execution(self, *dfs, trigger_hdk_import=False):
        results = [
            execute_pandas(df, modin_cfg=self.params, trigger_hdk_import=trigger_hdk_import)
            for df in dfs
        ]

        if len(dfs) == 1:
            return results[0]
        else:
            return results

    def trigger_execution(self, *dfs):
        """Utility function to trigger execution for lazy pd libraries. Returns actualized dfs.
        Some backends require separate method for data loading from disk, use `trigger_loading`
        for that."""
        return self._trigger_execution(*dfs, trigger_hdk_import=False)

    def trigger_loading(self, *dfs):
        """Trigger data loading for lazy libraries, should be called after reading data from disk."""
        return self._trigger_execution(*dfs, trigger_hdk_import=True)
