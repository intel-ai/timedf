import importlib

from pathlib import Path


def create_benchmark(bench_name):
    try:
        return importlib.import_module(f'.{bench_name}', __name__).Benchmark()
    except ModuleNotFoundError:
        available_benchmarks = [p.name for p in Path(__name__).iterdir() if p.is_dir()]
        raise ValueError(
            f'Attempted to create benchmark "{bench_name}", but it is missing from '
            f' the list of available benchmarkrs, which contains: "{available_benchmarks}"'
        )
