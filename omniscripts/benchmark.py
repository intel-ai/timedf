import abc
import argparse
from typing import Dict
import importlib


# # We keep benchmark import lazy to let the library initialize tool configuration before import
def create_benchmark(bench_name):
    # We are trying to dynamically import provided benchmark and want to catch
    # probelms with this import to report to user
    path = f"omniscripts_benchmarks.{bench_name}"
    try:
        return importlib.import_module(path, __name__).Benchmark()
    except ModuleNotFoundError as f:
        # The problem might be with some module that benchmark is using, like
        # missing `import os` inside of the benchmark, so check that before
        # creating error message
        if str(f) != f"No module named '{path}'":
            # Passthrough errors not directly related to benchmark module
            raise
        raise ValueError(
            f'Attempted to create benchmark "{bench_name}", but it is missing from '
            "the list of available benchmarks"
        )


class BenchmarkResults:
    def __init__(self, measurements: Dict[str, float], params=None) -> None:
        """Structure with benchmark results that is enforcing benchmark output format.

        Parameters
        ----------
        measurements
            Benchmark results in seconds in (query, time_s) form.
            Example: `{'load_data': 12.2, 'fe': 20.1}`
        params
            Additinal parameters of the current benchmark that need to be saved as well.
            Example: `{'dataset_size': 122, 'dfiles_n': 99}`
        """
        self._validate_dict(measurements)
        self._validate_vals(measurements, float)
        self.measurements = measurements
        self._validate_dict(params or {})
        self.params = self._convert_vals(params, str)

    @staticmethod
    def _validate_dict(res):
        if not isinstance(res, dict):
            raise ValueError(f"Measurements have to be of dict type, but they are {type(res)}")

    @staticmethod
    def _validate_vals(res, val_type):
        for key, val in res.items():
            if not isinstance(val, val_type):
                raise ValueError(f'Value for key="{key} is not {val_type}! type={type(val)}"')

    @staticmethod
    def _convert_vals(res, val_type):
        if res is None:
            return None
        return {k: val_type(v) for k, v in res.items()}


class BaseBenchmark(abc.ABC):
    # Tuple with parameters, specific for this benchmark
    __params__ = tuple()

    def add_benchmark_args(self, parser: argparse.ArgumentParser):
        """Benchmark can add arguments for parsing and they will be available in `params`
        during the run"""
        pass

    def run(self, params) -> BenchmarkResults:
        results = self.run_benchmark(params)
        if not isinstance(results, BenchmarkResults):
            raise ValueError(
                f"Benchmark must return instance of BenchmarkResults class, received {type(results)}"
            )

        return results

    @abc.abstractmethod
    def run_benchmark(self, params) -> BenchmarkResults:
        pass
