import os


def import_pandas_into_module_namespace(namespace, mode, ray_tmpdir=None, ray_memory=None):
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


def init_modin_on_hdk(pd):
    # Calcite initialization
    data = {"a": [1, 2, 3]}
    df = pd.DataFrame(data)
    df = df + 1
    _ = df.index
