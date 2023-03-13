import gc
import importlib

from omniscripts import BaseBenchmark, BenchmarkResults
from omniscripts.pandas_backend import set_backend

from .h2o_utils import tm, get_load_info, H2OBackend


def main_groupby(paths, backend):
    with tm.timeit("load_groupby_data"):
        df = backend.load_groupby_data(paths)

    with tm.timeit("groupby"):
        for name, q in backend.name2groupby_query.items():
            gc.collect()
            with tm.timeit(name):
                res = q(df)
                print(res)


def main_join(paths, backend):
    with tm.timeit("load_join_data"):
        data = backend.load_join_data(paths)

    with tm.timeit("join"):
        for name, q in backend.name2join_query.items():
            gc.collect()
            with tm.timeit(name):
                res = q(data)
                print(res)


def main(data_path, backend):
    with tm.timeit("total"):
        paths = get_load_info(data_path)
        main_groupby(paths, backend=backend)
        main_join(paths, backend=backend)


pandas_modes = ("Pandas", "Modin_on_ray", "Modin_on_hdk")
backend2impl = {
    "polars": "h2o_polars",
    **{n: "h2o_pandas" for n in pandas_modes},
}


class Benchmark(BaseBenchmark):
    def run_benchmark(self, params) -> BenchmarkResults:
        backend_name = params["pandas_mode"]

        if backend_name in pandas_modes:
            set_backend(backend_name, None, None)

        module = importlib.import_module(f"benchmarks.h2o_ext.{backend2impl[backend_name]}")
        backend: H2OBackend = module.H2OBackendImpl()

        main(data_path=params["data_file"], backend=backend)
        super().run_benchmark(params)
        measurement2time = tm.get_results()
        print(measurement2time)
        return BenchmarkResults(measurement2time)
