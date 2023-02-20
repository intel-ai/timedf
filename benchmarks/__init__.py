import importlib

from pathlib import Path

# Test the result
# Create PR


def create_benchmark(bench_name):
    try:
        benchmark = importlib.import_module(bench_name, ".").Benchmark()
    except ModuleNotFoundError as e:
        available_benchmarks = [p.name for p in Path(__name__).iterdir() if p.is_dir()]
        raise ValueError(
            f'Attempted to create benchmark "{bench_name}", but it is missing from '
            f' the list of available benchmarkrs, which contains: "{available_benchmarks}"'
        )
