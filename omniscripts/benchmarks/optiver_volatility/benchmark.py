from omniscripts import BenchmarkResults, BaseBenchmark

from .preprocess import preprocess
from .prepare_dataset import prepare_dataset
from .optiver_utils import get_workdir_paths, tm


def benchmark(paths):
    with tm.timeit("total"):
        with tm.timeit("01-preprocess"):
            preprocess(paths=paths)

        with tm.timeit("02-prepare_dataset_[ML&DP]"):
            prepare_dataset(paths=paths)


class Benchmark(BaseBenchmark):

    def run_benchmark(self, params) -> BenchmarkResults:
        paths = get_workdir_paths(params["data_file"])
        benchmark(paths=paths)

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)
