import abc


class BenchmarkResults:
    def __init__(self, measurements, params=None) -> None:
        """Results must be submited in seconds in a form {'q1': 12, 'q2': 13}, {'dataset_size': 122}

        Existing convention for benchmark results:
        - time is in seconds
        - Structure [{q1: x, q2: y}] what about notes?
        - No reporting of backend and iteration
        """
        self._validate_measurements(measurements)
        self.measurements = measurements
        self.params = params

    @staticmethod
    def _validate_measurements(measurements):
        return True


class BaseBenchmark(abc.ABC):
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
