import os
from typing import Union
from configparser import ConfigParser

import numpy as np
import psutil
import pandas as pd

from timedf.benchmark_utils import LaunchedProcesses

def import_pandas_into_module_namespace(
    namespace, mode, ray_tmpdir=None, ray_memory=None, num_threads=None
):
    def init_modin_on_hdk(pd):
        from modin.experimental.sql import query

        # Calcite initialization
        data = {"a": [1, 2, 3]}
        df = pd.DataFrame(data)
        query("SELECT * FROM df", df=df)

    if mode == "Pandas":
        print("Pandas backend: pure Pandas")
        import pandas as pd
    else:
        if num_threads:
            os.environ["MODIN_CPUS"] = str(num_threads)
        elif num_threads is None and os.environ.get("MODIN_CPUS", None) is not None:
            num_threads = int(os.environ["MODIN_CPUS"])
        if mode == "Modin_on_ray":
            import ray
            from modin.core.execution.utils import set_env

            if not ray_tmpdir:
                raise ValueError("`ray_tmpdir` wasn't provided")
            if not ray_memory:
                ray_memory = 200 * 1024 * 1024 * 1024
            if not ray.is_initialized():
                env_vars = {
                    "__MODIN_AUTOIMPORT_PANDAS__": "1",
                    "PYTHONWARNINGS": "ignore::FutureWarning",
                }
                with set_env(**env_vars):
                    ray.init(
                        include_dashboard=False,
                        _plasma_directory=ray_tmpdir,
                        _memory=ray_memory,
                        object_store_memory=ray_memory,
                        num_cpus=num_threads,
                    )
            os.environ["MODIN_ENGINE"] = "ray"
            print(
                f"Pandas backend: Modin on Ray with number of CPUs {num_threads}, tmp directory {ray_tmpdir} and memory {ray_memory}"
            )
        elif mode == "Modin_on_dask":
            os.environ["MODIN_ENGINE"] = "dask"
            print(f"Pandas backend: Modin on Dask with number of CPUs {num_threads}")
        elif mode == "Modin_on_python":
            os.environ["MODIN_ENGINE"] = "python"
            print(f"Pandas backend: Modin on pure Python with number of CPUs {num_threads}")
        elif mode == "Modin_on_hdk":
            os.environ["MODIN_ENGINE"] = "native"
            os.environ["MODIN_STORAGE_FORMAT"] = "hdk"
            os.environ["MODIN_EXPERIMENTAL"] = "True"
            print(f"Pandas backend: Modin on HDK with number of CPUs {num_threads}")
        elif mode == "Modin_on_unidist_mpi":
            os.environ["MODIN_ENGINE"] = "unidist"
            os.environ["UNIDIST_BACKEND"] = "mpi"

            import unidist

            if not unidist.config.MpiSpawn.get():
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                process_ids = comm.allgather(os.getpid())
                launched_process_list = [psutil.Process(pid) for pid in process_ids]
                LaunchedProcesses.get_instance().set_process_list(launched_process_list)

            if "MODIN_CPUS" in os.environ:
                os.environ["UNIDIST_CPUS"] = os.environ["MODIN_CPUS"]

            unidist.init()
            print(f"Pandas backend: Modin on Unidist with MPI with number of CPUs {num_threads}")
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


def trigger_import(df: pd.DataFrame):
    """
    Trigger import execution for DataFrame obtained by HDK engine.
    Parameters
    ----------
    df : DataFrame
        DataFrame for trigger import.
    """
    modin_frame = df._query_compiler._modin_frame
    if hasattr(modin_frame, "force_import"):
        modin_frame.force_import()
        return

    # The code below has been kept for backwards compatibility and will be removed in the future.

    from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import (
        DbWorker,
    )

    df.shape  # to trigger real execution

    p = modin_frame._partitions[0][0]
    if (
        p.frame_id is None
        and modin_frame._has_arrow_table()
        and not isinstance(table := p.get(), pd.DataFrame)
    ):
        p.frame_id = DbWorker().import_arrow_table(table)  # to trigger real execution


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
    if modin_cfg is None or isinstance(df, (pd.DataFrame, np.ndarray)):
        pass
    elif modin_cfg.StorageFormat.get().lower() == "hdk":
        if trigger_hdk_import:
            trigger_import(df)
        else:
            df._query_compiler._modin_frame._execute()
    else:
        df.shape
        partitions = df._query_compiler._modin_frame._partitions.flatten()
        mgr_cls = df._query_compiler._modin_frame._partition_mgr_cls
        if len(partitions) and hasattr(mgr_cls, "wait_partitions"):
            mgr_cls.wait_partitions(partitions)
        else:
            # compatibility with old Modin versions
            all(map(lambda partition: partition.drain_call_queue() or True, partitions))
            if modin_cfg.Engine.get().lower() == "ray":
                from ray import wait

                all(map(lambda partition: wait([partition._data]), partitions))
            elif modin_cfg.Engine.get().lower() == "dask":
                from dask.distributed import wait

                all(map(lambda partition: wait(partition._data), partitions))
            elif modin_cfg.Engine.get().lower() == "python":
                pass
    return df
