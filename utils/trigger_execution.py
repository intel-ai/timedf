import pandas


class Config:
    MODIN_IMPL = None
    MODIN_STORAGE_FORMAT = None
    MODIN_ENGINE = None

    @staticmethod
    def init(MODIN_IMPL, MODIN_STORAGE_FORMAT, MODIN_ENGINE):
        Config.MODIN_IMPL = MODIN_IMPL
        Config.MODIN_STORAGE_FORMAT = MODIN_STORAGE_FORMAT
        Config.MODIN_ENGINE = MODIN_ENGINE


def trigger_import(*dfs):
    """
    Trigger import execution for DataFrames obtained by HDK engine.
    Parameters
    ----------
    *dfs : iterable
        DataFrames to trigger import.
    """
    if Config.MODIN_STORAGE_FORMAT != "hdk" or Config.MODIN_IMPL == "pandas":
        return

    from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import (
        DbWorker,
    )

    for df in dfs:
        df.shape  # to trigger real execution
        df._query_compiler._modin_frame._partitions[0][0].frame_id = DbWorker().import_arrow_table(
            df._query_compiler._modin_frame._partitions[0][0].get()
        )  # to trigger real execution


def execute(df: pandas.DataFrame, trigger_hdk_import: bool = False):
    """
    Make sure the calculations are finished.
    Parameters
    ----------
    df : modin.pandas.DataFrame or pandas.Datarame
        DataFrame to be executed.
    trigger_hdk_import : bool, default: False
        Whether `df` are obtained by import with HDK engine.
    """
    df.shape
    if trigger_hdk_import:
        trigger_import(df)
        return

    if Config.MODIN_IMPL == "modin":
        if Config.MODIN_STORAGE_FORMAT == "hdk":
            df._query_compiler._modin_frame._execute()
            return

        partitions = df._query_compiler._modin_frame._partitions.flatten()
        if len(partitions) > 0 and hasattr(partitions[0], "wait"):
            all(map(lambda partition: partition.wait(), partitions))
            return

        # compatibility with old Modin versions
        all(map(lambda partition: partition.drain_call_queue() or True, partitions))
        if Config.MODIN_ENGINE == "ray":
            from ray import wait

            all(map(lambda partition: wait([partition._data]), partitions))
        elif Config.MODIN_ENGINE == "dask":
            from dask.distributed import wait

            all(map(lambda partition: wait(partition._data), partitions))
        elif Config.MODIN_ENGINE == "python":
            pass

    elif Config.MODIN_IMPL == "pandas":
        pass
    else:
        raise ValueError(f"Unknown modin implementation used {Config.MODIN_IMPL}")


def trigger_execution(*dfs):
    """Utility function to trigger execution for lazy pd libraries."""
    for df in dfs:
        execute(df)
