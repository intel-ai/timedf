import importlib
from pathlib import Path

from omniscripts.benchmark import BaseBenchmark


# We keep benchmark import lazy to let the library initialize tool configuration before import
def create_benchmark(bench_name) -> BaseBenchmark:
    # We are trying to dynamically import provided benchmark and want to catch
    # probelms with this import to report to user
    try:
        return importlib.import_module(f".{bench_name}", __name__).Benchmark()
    except ModuleNotFoundError as f:
        # The problem might be with some module that benchmark is using, like
        # missing `import os` inside of the benchmark, so check that before
        # creating error message
        if str(f) != f"No module named '{bench_name}'":
            # Passthrough errors not directly related to benchmark module
            raise
        available_benchmarks = [p.name for p in Path(__name__).iterdir() if p.is_dir()]
        raise ValueError(
            f'Attempted to create benchmark "{bench_name}", but it is missing from '
            f'the list of available benchmarkrs, which contains: "{available_benchmarks}"'
        )
