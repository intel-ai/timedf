from pathlib import Path

from utils import BenchmarkResults, BaseBenchmark, check_support
from .preprocess import preprocess
from .prepare_dataset import prepare_dataset
from .optiver_utils import get_workdir_paths, tm


def benchmark(paths):
    with tm.timeit("01-preprocess"):
        preprocess(raw_data_path=paths['raw_data'], preprocessed_path=paths["preprocessed"])

    with tm.timeit("02-prepare dataset"):
        prepare_dataset(paths=paths)


class Benchmark(BaseBenchmark):
    def run_benchmark(self, parameters) -> BenchmarkResults:
        check_support(parameters, unsupported_params=["optimizer", "dfiles_num"])

        raw_data_path = Path(parameters["data_file"].strip("'"))
        paths = get_workdir_paths(raw_data_path)
        benchmark(paths=paths)

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)
