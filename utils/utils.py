import glob
import os
import time
import warnings
from timeit import default_timer as timer
from tempfile import mkstemp

import psutil

from report import BenchmarkDb
from benchmarks import create_benchmark
from utils_base_env import DbConfig

from .namespace_utils import import_pandas_into_module_namespace
from .pandas_backend import set_backend
from .benchmark import BaseBenchmark, BenchmarkResults


repository_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
directories = {"repository_root": repository_root_directory}


def load_data_pandas(
    filename,
    columns_names=None,
    columns_types=None,
    header=None,
    nrows=None,
    use_gzip=False,
    parse_dates=None,
    pd=None,
    pandas_mode="Pandas",
):
    if not pd:
        import_pandas_into_module_namespace(
            namespace=load_data_pandas.__globals__, mode=pandas_mode
        )
    types = None
    if columns_types:
        types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    return pd.read_csv(
        filename,
        names=columns_names,
        nrows=nrows,
        header=header,
        dtype=types,
        compression="gzip" if use_gzip else None,
        parse_dates=parse_dates,
    )


def load_data_modin_on_hdk(
    filename, columns_names=None, columns_types=None, parse_dates=None, pd=None, skiprows=None
):
    if not pd:
        import_pandas_into_module_namespace(
            namespace=load_data_pandas.__globals__, mode="Modin_on_hdk"
        )
    dtypes = None
    if columns_types:
        dtypes = {
            columns_names[i]: columns_types[i] if (columns_types[i] != "category") else "string"
            for i in range(len(columns_names))
        }

    all_but_dates = dtypes
    dates_only = False
    if parse_dates:
        parse_dates = parse_dates if isinstance(parse_dates, (list, tuple)) else [parse_dates]
        all_but_dates = {
            col: valtype for (col, valtype) in dtypes.items() if valtype not in parse_dates
        }
        dates_only = [col for (col, valtype) in dtypes.items() if valtype in parse_dates]

    return pd.read_csv(
        filename,
        names=columns_names,
        dtype=all_but_dates,
        parse_dates=dates_only,
        skiprows=skiprows,
    )


def expand_braces(pattern: str):
    """
    Expand braces of the provided string in Linux manner.

    `pattern` should be passed in the next format:
    pattern = "prefix{values_to_expand}suffix"

    Notes
    -----
    `braceexpand` replacement for single string format type.
    Can be used to avoid package import for single corner
    case.

    Examples
    --------
    >>> expand_braces("/taxi/trips_xa{a,b,c}.csv")
    ['/taxi/trips_xaa.csv', '/taxi/trips_xab.csv', '/taxi/trips_xac.csv']
    """
    brace_open_idx = pattern.index("{")
    brace_close_idx = pattern.index("}")

    prefix = pattern[:brace_open_idx]
    suffix = pattern[brace_close_idx + 1 :]
    choices = pattern[brace_open_idx + 1 : brace_close_idx].split(",")

    expanded = []
    for choice in choices:
        expanded.append(prefix + choice + suffix)

    return expanded


def files_names_from_pattern(files_pattern):
    try:
        from braceexpand import braceexpand
    except ModuleNotFoundError:
        braceexpand = None

    data_files_names = None
    path_expander = glob.glob
    data_files_names = (
        list(braceexpand(files_pattern)) if braceexpand else expand_braces(files_pattern)
    )

    if "://" in files_pattern:
        from .s3_client import s3_client

        if all(map(s3_client.s3like, data_files_names)):
            path_expander = s3_client.glob
        else:
            raise ValueError(f"some of s3like links are bad: {data_files_names}")

    return sorted([x for f in data_files_names for x in path_expander(f)])


def print_results(results, backend=None, ignore_fields=[]):
    if backend:
        print(f"{backend} results:")
    for result_name, result in results.items():
        if result_name not in ignore_fields:
            print("    {} = {:.3f} {}".format(result_name, result, 's'))


# SklearnImport imports sklearn (intel or stock version) only if it is not done previously
class SklearnImport:
    def __init__(self):
        self.current_optimizer = None
        self.train_test_split = None

    def get_train_test_split(self, optimizer):
        assert optimizer is not None, "optimizer parameter should be specified"
        if self.current_optimizer is not optimizer:
            self.current_optimizer = optimizer

            if optimizer == "intel":
                import sklearnex

                sklearnex.patch_sklearn()
                from sklearn.model_selection import train_test_split
            elif optimizer == "stock":
                from sklearn.model_selection import train_test_split
            else:
                raise ValueError(
                    f"Intel optimized and stock sklearn are supported. \
                    {optimizer} can't be recognized"
                )
            self.train_test_split = train_test_split

        return self.train_test_split


sklearn_import = SklearnImport()


def split(X, y, test_size=0.1, stratify=None, random_state=None, optimizer="intel"):
    train_test_split = sklearn_import.get_train_test_split(optimizer)

    t0 = timer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    split_time = timer() - t0

    return (X_train, y_train, X_test, y_test), split_time


def check_support(current_params, unsupported_params):
    ignored_params = {}
    for param in unsupported_params:
        if current_params.get(param) is not None:
            ignored_params[param] = current_params[param]

    if ignored_params:
        warnings.warn(f"Parameters {ignored_params} are ignored", RuntimeWarning)


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)  # GB units


def getsize(filename: str):
    """Return size of filename in MB"""
    if "://" in filename:
        from .s3_client import s3_client

        if s3_client.s3like(filename):
            return s3_client.getsize(filename) / 1024 / 1024
        raise ValueError(f"bad s3like link: {filename}")
    else:
        return os.path.getsize(filename) / 1024 / 1024


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

    # Set current backend, !!!needs to be run before benchmark import!!!
    set_backend(pandas_mode=pandas_mode, ray_tmpdir=ray_tmpdir, ray_memory=ray_memory)

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

    reporter = db_config and BenchmarkDb(db_config.create_engine())

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
