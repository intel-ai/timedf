import argparse

import numpy as np

# No pandas is needed, it is imported on the next line
# This is optional, this import will provide you with either pandas or modin, depending on run
# params. If you not working with pandas API you have no point in using it.
# However, if you are measuring pandas against pandas API library like modin you are advised to use
# this tool.
from omniscripts.pandas_backend import pd

from omniscripts import BaseBenchmark, BenchmarkResults
from omniscripts import TimerManager


# This is an optional tool to simplity time tracking, without it you need to gather timing results
# manually into a simple dict (measurement: time_s) like {'load_data': 11.1, 'fe': 1.122}
tm = TimerManager()


class Benchmark(BaseBenchmark):
    # You can pass benchmark-specific parameters by creating this field and
    # writing `add_benchmark_args` function
    __params__ = ("my_param",)

    # Parse your benchmark-specific arguments
    def add_benchmark_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-my_param",
            default="my_value",
            help="You can pass your own value this way.",
        )

    def run_benchmark(self, params) -> BenchmarkResults:
        N = 10_000

        print("My param is ", params["my_param"])

        # This command will measure time it took to run nested block and record it with
        # 'load_data' key
        with tm.timeit("load_data"):
            # your data path will be in params["data_file"]
            # so you can use pd.read_csv(params["data_file"]) to read it
            df = pd.DataFrame(np.random.randint(0, 100, size=(N, 4)), columns=list("abcd"))

        with tm.timeit("fe"):
            df["f"] = (df["a"] ** 2 + df["b"] ** 2) ** 0.5

        with tm.timeit("evaluation"):
            df["y"] = (df["a"] + df["c"] + df["d"] + df["f"]) / 4 > 0.5

        # This will be a dict with timing results that look like:
        # {'load_data': 1.1, 'fe': 11.1, 'evaluation': 1.12}
        measurement2time = tm.get_results()

        # This is optional, you can pass arbitrary json data to save along with benchmark results
        benchmark_specific_record = {"dataset_size": N}
        # This is required result format
        return BenchmarkResults(measurement2time, benchmark_specific_record)
