import os
from typing import Union
from configparser import ConfigParser

import numpy as np
import pandas as pd


def import_pandas_into_module_namespace(namespace, mode, ray_tmpdir=None, ray_memory=None):
    def init_modin_on_hdk(pd):
        # Calcite initialization
        data = {"a": [1, 2, 3]}
        df = pd.DataFrame(data)
        df = df + 1
        _ = df.index

    if mode == "Pandas":
        print("Pandas backend: pure Pandas")
        import pandas as pd
    else:
        if mode == "Modin_on_ray":
            import ray

            if not ray_tmpdir:
                ray_tmpdir = "/tmp"
            if not ray_memory:
                ray_memory = 200 * 1024 * 1024 * 1024
            if not ray.is_initialized():
                ray.init(
                    include_dashboard=False,
                    _plasma_directory=ray_tmpdir,
                    _memory=ray_memory,
                    object_store_memory=ray_memory,
                )
            os.environ["MODIN_ENGINE"] = "ray"
            print(
                f"Pandas backend: Modin on Ray with tmp directory {ray_tmpdir} and memory {ray_memory}"
            )
        elif mode == "Modin_on_dask":
            os.environ["MODIN_ENGINE"] = "dask"
            print("Pandas backend: Modin on Dask")
        elif mode == "Modin_on_python":
            os.environ["MODIN_ENGINE"] = "python"
            print("Pandas backend: Modin on pure Python")
        elif mode == "Modin_on_hdk":
            os.environ["MODIN_ENGINE"] = "native"
            os.environ["MODIN_STORAGE_FORMAT"] = "hdk"
            os.environ["MODIN_EXPERIMENTAL"] = "True"
            print("Pandas backend: Modin on HDK")
        elif mode == "Modin_on_Unidist_mpi":
            os.environ["MODIN_ENGINE"] = "unidist"
            os.environ["UNIDIST_BACKEND"] = "mpi"
            if 'MODIN_CPUS' in os.environ:
                os.environ["UNIDIST_CPUS"] = os.environ["MODIN_CPUS"]
            import unidist
            unidist.init()
        else:
            raise ValueError(f"Unknown pandas mode {mode}")
        import modin.pandas as pd

        # Some components of Modin on HDK engine are initialized only
        # at the moment of query execution, so for proper benchmarks performance
        # measurement we need to initialize these parts before any measurements
        if mode == "Modin_on_hdk":
            init_modin_on_hdk(pd)
    if not isinstance(namespace, (list, tuple)):
        namespace = [namespace]
    for space in namespace:
        space["pd"] = pd


def trigger_import(*dfs):
    """
    Trigger import execution for DataFrames obtained by HDK engine.
    Parameters
    ----------
    *dfs : iterable
        DataFrames to trigger import.
    """
    from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import (
        DbWorker,
    )

    for df in dfs:
        df.shape  # to trigger real execution
        df._query_compiler._modin_frame._partitions[0][0].frame_id = DbWorker().import_arrow_table(
            df._query_compiler._modin_frame._partitions[0][0].get()
        )  # to trigger real execution


def execute(
    df: pd.DataFrame, *, trigger_hdk_import: bool = False, modin_cfg: Union[None, ConfigParser]
):
    """Make sure the calculations are finished.

    Parameters
    ----------
    df : modin.pandas.DataFrame or pandas.Datarame
        DataFrame to be executed.
    trigger_hdk_import : bool, default: False
        Whether `df` are obtained by import with HDK engine.
    modin_cfg: modin config
        Modin configuration that defines values for `StorageFormat` and `Engine`.
        If None, pandas backend is assumed.
    """
    if modin_cfg is None:
        return

    df.shape

    if isinstance(df, (pd.DataFrame, np.ndarray)):
        return

    if trigger_hdk_import and modin_cfg.StorageFormat.get() == "hdk":
        trigger_import(df, modin_cfg=modin_cfg)

    if modin_cfg.StorageFormat.get() == "hdk":
        df._query_compiler._modin_frame._execute()
        return

    partitions = df._query_compiler._modin_frame._partitions.flatten()
    mgr_cls = df._query_compiler._modin_frame._partition_mgr_cls
    if len(partitions) and hasattr(mgr_cls, "wait_partitions"):
        mgr_cls.wait_partitions(partitions)
        return

    # compatibility with old Modin versions
    all(map(lambda partition: partition.drain_call_queue() or True, partitions))
    if modin_cfg.Engine.get() == "ray":
        from ray import wait

        all(map(lambda partition: wait([partition._data]), partitions))
    elif modin_cfg.Engine.get() == "dask":
        from dask.distributed import wait

        all(map(lambda partition: wait(partition._data), partitions))
    elif modin_cfg.Engine.get() == "python":
        pass


def trigger_execution_base(*dfs, modin_cfg=Union[None, ConfigParser]):
    """Utility function to trigger execution for lazy pd libraries.

    Parameters
    ----------
    dfs:
        Dataframes or numpy arrays to materialize
    modin_cfg:
        Modin configuration that defines values for `StorageFormat` and `Engine`.
        If None, pandas backend is assumed.
    """
    for df in dfs:
        execute(df, modin_cfg=modin_cfg)
