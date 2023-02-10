from pathlib import Path

from utils import BenchmarkResults, BaseBenchmark, check_support
from .preprocess import preprocess
from .optiver_utils import get_workdir_paths, tm


class Benchmark(BaseBenchmark):
    def run_benchmark(self, parameters) -> BenchmarkResults:
        check_support(parameters, unsupported_params=["optimizer", "dfiles_num"])

        raw_data_path = Path(parameters["data_file"].strip("'"))

        paths = get_workdir_paths()
        with tm.timeit('preprocess'):
            preprocess(raw_data_path=raw_data_path, preprocessed_path=paths['preprocessed'])

        task2time = tm.get_results()
        print(task2time)

        return BenchmarkResults(task2time)