import abc
import time
import warnings
from typing import Dict

from .pandas_backend import set_backend

from env_manager import DbConfig


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
    # Unsupported running parameters to warn user about
    __unsupported_params__ = tuple()

    def prerun(self, params):
        self.check_support(params)

    def run(self, params) -> BenchmarkResults:
        self.prerun(params)
        results = self.run_benchmark(params)
        if not isinstance(results, BenchmarkResults):
            raise ValueError(
                f"Benchmark must return instance of BenchmarkResults class, received {type(results)}"
            )

        return results

    def check_support(self, params):
        ignored_params = {}
        for param in self.__unsupported_params__:
            if params.get(param) is not None:
                ignored_params[param] = params[param]

        if ignored_params:
            warnings.warn(f"Parameters {ignored_params} are ignored", RuntimeWarning)

    @abc.abstractmethod
    def run_benchmark(self, params) -> BenchmarkResults:
        pass


def run_benchmarks(
    bench_name: str,
    data_file: str,
    dfiles_num: int = None,
    iterations: int = 1,
    validation: bool = False,
    optimizer: str = None,
    pandas_mode: str = "Pandas",
    ray_tmpdir: str = "/tmp",
    ray_memory: int = 200 * 1024 * 1024 * 1024,
    no_ml: bool = None,
    use_modin_xgb: bool = False,
    gpu_memory: int = None,
    extended_functionality: bool = False,
    db_config: DbConfig = None,
    commit_hdk: str = "1234567890123456789012345678901234567890",
    commit_omniscripts: str = "1234567890123456789012345678901234567890",
    commit_modin: str = "1234567890123456789012345678901234567890",
):
    """
    Run benchmarks for Modin perf testing and report results.

    Parameters
    ----------
    bench_name : str
        Benchmark name.
    data_file : str
        A datafile that should be loaded.
    dfiles_num : int, optional
        Number of datafiles to load into database for processing.
    iterations : int, default: 1
        Number of iterations to run every query. The best result is selected.
    validation : bool, default: False
        Validate queries results (by comparison with Pandas queries results).
    optimizer : str, optional
        Optimizer to use.
    pandas_mode : str, default: "Pandas"
        Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask or on HDK.
    ray_tmpdir : str, default: "/tmp"
        Location where to keep Ray plasma store. It should have enough space to keep `ray_memory`.
    ray_memory : int, default: 200 * 1024 * 1024 * 1024
        Size of memory to allocate for Ray plasma store.
    no_ml : bool, optional
        Do not run machine learning benchmark, only ETL part.
    use_modin_xgb : bool, default: False
        Whether to use Modin XGBoost for ML part, relevant for Plasticc benchmark only.
    gpu_memory : int, optional
        Specify the memory of your gpu(This controls the lines to be used. Also work for CPU version).
    extended_functionality : bool, default: False
        Extends functionality of H2O benchmark by adding 'chk' functions and verbose local reporting of results.
    db_config: DbConfig, optional
        Configuration for the database
    commit_hdk : str, default: "1234567890123456789012345678901234567890"
        HDK commit hash used for benchmark.
    commit_omniscripts : str, default: "1234567890123456789012345678901234567890"
        Omniscripts commit hash used for benchmark.
    commit_modin : str, default: "1234567890123456789012345678901234567890"
        Modin commit hash used for benchmark.
    """

    data_file = data_file.replace("'", "")

    # Set current backend, !!!needs to be run before benchmark import!!!
    set_backend(pandas_mode=pandas_mode, ray_tmpdir=ray_tmpdir, ray_memory=ray_memory)

    from benchmarks import create_benchmark

    benchmark: BaseBenchmark = create_benchmark(bench_name)

    run_parameters = {
        "data_file": data_file,
        "dfiles_num": dfiles_num,
        "no_ml": no_ml,
        "use_modin_xgb": use_modin_xgb,
        "optimizer": optimizer,
        "pandas_mode": pandas_mode,
        "ray_tmpdir": ray_tmpdir,
        "ray_memory": ray_memory,
        "gpu_memory": gpu_memory,
        "validation": validation,
        "extended_functionality": extended_functionality,
        "commit_hdk": commit_hdk,
        "commit_omniscripts": commit_omniscripts,
        "commit_modin": commit_modin,
    }

    run_id = int(round(time.time()))
    print(run_parameters)

    if db_config is not None:
        from .report import BenchmarkDb

        reporter = BenchmarkDb(db_config.create_engine())
    else:
        reporter = None

    for iter_num in range(1, iterations + 1):
        print(f"Iteration #{iter_num}")
        results: BenchmarkResults = benchmark.run(run_parameters)

        if reporter is not None:
            reporter.report(
                iteration_no=iter_num,
                name2time=results.measurements,
                params=results.params,
                benchmark=bench_name,
                run_id=run_id,
                run_params=run_parameters,
            )
