import glob
import os
import time
from typing import Dict, List
import warnings
from dataclasses import dataclass
from timeit import default_timer as timer
from collections import OrderedDict
from tempfile import mkstemp

import psutil

from .namespace_utils import import_pandas_into_module_namespace
from .pandas_backend import set_backend
from utils_base_env.benchmarks import benchmark_mapper


conversions = {"ms": 1000, "s": 1, "m": 1 / 60, "": 1}
repository_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
directories = {"repository_root": repository_root_directory}
ny_taxi_data_files_sizes_MB = OrderedDict(
    {
        "trips_xaa.csv": 8000,
        "trips_xab.csv": 8100,
        "trips_xac.csv": 4200,
        "trips_xad.csv": 7300,
        "trips_xae.csv": 8600,
        "trips_xaf.csv": 8600,
        "trips_xag.csv": 8600,
        "trips_xah.csv": 8600,
        "trips_xai.csv": 8600,
        "trips_xaj.csv": 8600,
        "trips_xak.csv": 8700,
        "trips_xal.csv": 8700,
        "trips_xam.csv": 8600,
        "trips_xan.csv": 8600,
        "trips_xao.csv": 8600,
        "trips_xap.csv": 8600,
        "trips_xaq.csv": 8600,
        "trips_xar.csv": 8600,
        "trips_xas.csv": 8600,
        "trips_xat.csv": 8600,
    }
)


def get_percentage(error_message):
    # parsing message like: lalalalal values are different (xxxxx%) lalalalal
    return float(error_message.split("values are different ")[1].split("%)")[0][1:])


def compare_columns(columns):
    if len(columns) != 2:
        raise AttributeError(f"Columns number should be 2, actual number is {len(columns)}")

    import pandas as pd

    # in percentage - 0.05 %
    max_error = 0.05

    try:
        pd.testing.assert_series_equal(
            columns[0],
            columns[1],
            check_less_precise=2,
            check_dtype=False,
            check_categorical=False,
        )
        if str(columns[0].dtype) == "category":
            left = columns[0]
            right = columns[1]
            assert left.cat.ordered == right.cat.ordered
            # assert_frame_equal cannot turn off comparison of
            # order of categories, so compare categories manually
            pd.testing.assert_series_equal(
                left,
                right,
                check_dtype=False,
                check_less_precise=2,
                check_category_order=left.cat.ordered,
            )
    except AssertionError as assert_err:
        if str(columns[0].dtype).startswith("float"):
            try:
                current_error = get_percentage(str(assert_err))
                if current_error > max_error:
                    print(
                        f"Max acceptable difference: {max_error}%; current difference: {current_error}%"
                    )
                    raise assert_err
            # for catch exceptions from `get_percentage`
            except Exception:
                raise assert_err
        else:
            raise


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


def print_times(times, backend=None):
    if backend:
        print(f"{backend} times:")
    for time_name, _time in times.items():
        print("{} = {:.5f} s".format(time_name, _time))


def print_results(results, backend=None, unit="", ignore_fields=[]):
    results_converted = convert_units(results, ignore_fields=[], unit=unit)
    if backend:
        print(f"{backend} results:")
    for result_name, result in results_converted.items():
        if result_name not in ignore_fields:
            print("    {} = {:.3f} {}".format(result_name, result, unit))


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


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


def timer_ms():
    return round(timer() * 1000)


def remove_fields_from_dict(dictonary, fields_to_remove):
    for key in fields_to_remove or ():
        dictonary.pop(key, None)


def convert_units(dict_to_convert, ignore_fields, unit="ms"):
    try:
        multiplier = conversions[unit]
    except KeyError:
        raise ValueError(f"Conversion to {unit} is not implemented")

    return {
        key: (value * multiplier if key not in ignore_fields else value)
        for key, value in dict_to_convert.items()
    }


def check_fragments_size(fragments_size, count_table, default_fragments_size=None):
    result_fragments_size = []

    if fragments_size:
        result_fragments_size = fragments_size
    elif default_fragments_size:
        result_fragments_size = default_fragments_size
    else:
        result_fragments_size = [None] * count_table

    return result_fragments_size


def write_to_csv_by_chunks(file_to_write, output_file, write_mode="wb", chunksize=1024):
    import zlib

    wbits_gzip = 16 + zlib.MAX_WBITS

    with open(file_to_write, "rb") as f:
        buffer = f.read(chunksize)

        if file_to_write.endswith(".gz"):
            d = zlib.decompressobj(wbits=wbits_gzip)
            while buffer:
                # Some of the input data may be preserved in internal buffers for later processing
                # so we should use `flush` at the end of processing
                chunk = d.decompress(buffer)
                with open(output_file, write_mode) as output:
                    output.write(chunk)
                buffer = f.read(chunksize)

            chunk = d.flush()
            with open(output_file, write_mode) as output:
                output.write(chunk)
        elif file_to_write.endswith(".csv"):
            while buffer:
                with open(output_file, write_mode) as output:
                    output.write(buffer)
                buffer = f.read(chunksize)
        else:
            raise NotImplementedError(f"file' extension: [{file_to_write}] is not supported yet")


def check_support(current_params, unsupported_params):
    ignored_params = {}
    for param in unsupported_params:
        if current_params.get(param) is not None:
            ignored_params[param] = current_params[param]

    if ignored_params:
        warnings.warn(f"Parameters {ignored_params} are ignored", RuntimeWarning)


def create_dir(dir_name):
    directory = os.path.abspath(os.path.join(directories["repository_root"], dir_name))
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def get_ny_taxi_dataset_size(dfiles_num):
    return sum(list(ny_taxi_data_files_sizes_MB.values())[:dfiles_num])


def make_chk(values):
    s = ";".join(str_round(x) for x in values)
    return s.replace(",", "_")  # comma is reserved for csv separator


def str_round(x):
    if type(x).__name__ in ["float", "float64"]:
        x = round(x, 3)
    return str(x)


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


def join_to_tbls(data_name):
    """Prepare H2O join queries data files (for merge right parts) names basing on the merge left data file name.

    Parameters
    ----------
    data_name: str
        Merge left data file name, should contain "NA" component.

    Returns
    -------
    data_files_paths: dict
        Dictionary with data files paths, dictionary keys: "x", "small", "medium", "big".
    data_files_sizes: dict
        Dictionary with data files sizes, dictionary keys: "x", "small", "medium", "big".

    """
    data_dir = os.path.dirname(os.path.abspath(data_name))
    data_file = data_name.replace(data_dir, "")
    x_n = int(float(data_file.split("_")[1]))
    y_n = ["{:.0e}".format(x_n / 1e6), "{:.0e}".format(x_n / 1e3), "{:.0e}".format(x_n)]
    y_n = [y.replace("+0", "") for y in y_n]
    y_n = [data_name.replace("NA", y) for y in y_n]
    data_files_paths = {"x": data_name, "small": y_n[0], "medium": y_n[1], "big": y_n[2]}
    data_files_sizes = {
        data_id: getsize(data_file) for data_id, data_file in data_files_paths.items()
    }
    return data_files_paths, data_files_sizes


def get_tmp_filepath(filename, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = create_dir("tmp")

    filename, extension = os.path.splitext(filename)

    # filename would be transormed like "census-fsi.csv" -> "ROOT_RESOPOSITORY_DIR/tmp/census-fsi-f15cxc9y.csv"
    file_descriptor, file_path = mkstemp(suffix=extension, prefix=filename + "-", dir=tmp_dir)
    os.close(file_descriptor)

    return file_path


class FilesCombiner:
    """
    If data files are compressed or number of csv files is more than one,
    data files (or single compressed file) should be transformed to single csv file.
    Before files transformation, script checks existance of already transformed file
    in the directory passed with -data_file flag.
    """

    def __init__(self, data_files_names, combined_filename, files_limit):
        self._data_files_names = data_files_names
        self._files_limit = files_limit

        data_file_path = self._data_files_names[0]

        _, data_files_extension = os.path.splitext(data_file_path)
        if data_files_extension == ".gz" or len(data_files_names) > 1:
            data_file_path = os.path.abspath(
                os.path.join(os.path.dirname(data_files_names[0]), combined_filename)
            )

        self._should_combine = not os.path.exists(data_file_path)
        if self._should_combine:
            data_file_path = get_tmp_filepath(combined_filename)

        self._data_file_path = data_file_path

    def __enter__(self):
        if self._should_combine:
            for file_name in self._data_files_names[: self._files_limit]:
                write_to_csv_by_chunks(
                    file_to_write=file_name, output_file=self._data_file_path, write_mode="ab"
                )

        return self._data_file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_combine:
            try:
                os.remove(self._data_file_path)
            except FileNotFoundError:
                pass


def refactor_results_for_reporting(
    benchmark_results: dict,
    ignore_fields_for_results_unit_conversion: list = None,
    additional_fields: dict = None,
    reporting_unit: str = "ms",
) -> dict:

    """Refactore benchmarks results in the way they can be easily reported to MySQL database.

    Parameters
    ----------
    benchmark_results: dict
        Dictionary with results reported by benchmark.
        Dictionary should follow the next pattern: {"ETL": [<dicts_with_etl_results>], "ML": [<dicts_with_ml_results>]}.
    ignore_fields_for_results_unit_conversion: list
        List of fields that should be ignored during results unit conversion.
    additional_fields: dict
        Dictionary with fields that should be additionally reported to MySQL database.
        Dictionary should follow the next pattern: {"ETL": {<dicts_with_etl_fields>}, "ML": {<dicts_with_ml_fields>}}.
    reporting_unit: str
        Time unit name for results reporting to MySQL database. Accepted values are "ms", "s", "m".

    Return
    ------
    etl_ml_results: dict
        Refactored benchmark results for reporting to MySQL database.
        Dictionary follows the next pattern: {"ETL": [<etl_results>], "ML": [<ml_results>]}

    """

    etl_ml_results = {"ETL": [], "ML": []}
    for results_category, results in dict(benchmark_results).items():  # ETL or ML part
        for backend_result in results:
            if backend_result is not None and all(
                isinstance(r, dict) for r in backend_result.values()
            ):  # True if subqueries are used
                backend_result_converted = []
                for query_name, query_results in backend_result.items():
                    query_results.update({"query_name": query_name})
                    backend_result_converted.append(query_results)
            else:
                backend_result_converted = [backend_result]

            for result in backend_result_converted:
                if result:
                    result = convert_units(
                        result,
                        ignore_fields=ignore_fields_for_results_unit_conversion,
                        unit=reporting_unit,
                    )
                    result.update(additional_fields.get(results_category, {}))
                    etl_ml_results[results_category].append(result)

    return etl_ml_results


def get_dir_size(start_path="."):
    """Get directory size including all subdirectories.

    Parameters
    ----------
    start_path:
        Path to begin calculation of directory size.

    Return
    ------
    total_size:
        Total size of directory in MB.

    """
    total_size = 0
    if "://" in start_path:
        from .s3_client import s3_client

        if s3_client.s3like(start_path):
            total_size = s3_client.du(start_path)
        else:
            raise ValueError(f"bad s3like link: {start_path}")
    else:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += getsize(fp)

    return total_size


@dataclass
class DBParams:
    driver: str
    server: str
    port: int
    user: str
    password: str
    name: str


class ResultReporter:
    def __init__(self, db_params: DBParams, table_name, predefined_fields, ignore_fields) -> None:
        self.db_params = db_params
        self.table_name = table_name
        self.predefined_fields = predefined_fields
        self.ignore_fields = ignore_fields
        self.reporter = None

    def _get_db_url(self):
        return f"{self.db_params.driver}://{self.db_params.user}:{self.db_params.password}@{self.db_params.server}:{self.db_params.port}/{self.db_params.name}"

    def _initialize_report(self, results):
        """This function exists because currently we need the first result to talk to the database"""
        from report import DbReport
        from sqlalchemy import create_engine

        self.engine = create_engine(self._get_db_url(), future=True)
        benchmark_fields = list({val for result_row in results for val in result_row})

        self.reporter = DbReport(
            self.engine, self.table_name, benchmark_fields, self.predefined_fields
        )

    def report(self, results: List[Dict[str, float]]):
        if self.reporter is None and len(results) > 0:
            self._initialize_report(results)

        for result in results:
            remove_fields_from_dict(result, self.ignore_fields)
            self.reporter.submit(result)


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
    db_driver: str = "mysql+mysqlconnector",
    db_server: str = "localhost",
    db_port: int = 3306,
    db_user: str = None,
    db_pass: str = "omniscidb",
    db_name: str = "omniscidb",
    db_table_etl: str = None,
    db_table_ml: str = None,
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
    db_server : str, default: "localhost"
        Host name of MySQL server.
    db_port : int, default: 3306
        Port number of MySQL server.
    db_user : str, optional
        Username to use to connect to MySQL database. If user name is specified, script attempts to store
        results in MySQL database using other `db-*` parameters.
    db_pass : str, default: "omniscidb"
        Password to use to connect to MySQL database.
    db_name : str, default: "omniscidb"
        MySQL database to use to store benchmark results.
    db_table_etl : str, optional
        Table to store ETL results for this benchmark.
    db_table_ml : str, optional
        Table to store ML results for this benchmark.
    commit_hdk : str, default: "1234567890123456789012345678901234567890"
        HDK commit hash used for benchmark.
    commit_omniscripts : str, default: "1234567890123456789012345678901234567890"
        Omniscripts commit hash used for benchmark.
    commit_modin : str, default: "1234567890123456789012345678901234567890"
        Modin commit hash used for benchmark.
    """
    ignore_fields_for_results_unit_conversion = [
        "Backend",
        "dfiles_num",
        "dataset_size",
        "query_name",
    ]

    # Set current backend, !!!needs to be run before benchmark import!!!
    set_backend(pandas_mode=pandas_mode, ray_tmpdir=ray_tmpdir, ray_memory=ray_memory)

    run_benchmark = __import__(benchmark_mapper[bench_name]).run_benchmark

    parameters = {
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
    }

    db_params = DBParams(
        driver=db_driver,
        server=db_server,
        port=db_port,
        user=db_user,
        password=db_pass,
        name=db_name,
    )
    predefined_fields = {
        "OmnisciCommitHash": commit_hdk,
        "OmniscriptsCommitHash": commit_omniscripts,
        "ModinCommitHash": commit_modin,
    }
    etl_reporter = ResultReporter(db_params, db_table_etl, predefined_fields, ["t_connect"])
    ml_reporter = ResultReporter(db_params, db_table_ml, predefined_fields, [])

    print(parameters)
    run_id = int(round(time.time()))

    for iter_num in range(1, iterations + 1):
        print(f"Iteration #{iter_num}")
        benchmark_results = run_benchmark(parameters)

        # This part is necessary because some benchmarks miss ML part. It's a temporary solution,
        # in the long run it's better to implement https://github.com/intel-ai/omniscripts/issues/317
        if "ML" not in benchmark_results:
            benchmark_results["ML"] = []

        additional_fields_for_reporting = {
            "ETL": {"Iteration": iter_num, "run_id": run_id},
            "ML": {"Iteration": iter_num, "run_id": run_id},
        }
        etl_ml_results = refactor_results_for_reporting(
            benchmark_results=benchmark_results,
            ignore_fields_for_results_unit_conversion=ignore_fields_for_results_unit_conversion,
            additional_fields=additional_fields_for_reporting,
            reporting_unit="ms",
        )
        etl_results = list(etl_ml_results["ETL"])
        ml_results = list(etl_ml_results["ML"])

        if db_user is not None:
            etl_reporter.report(etl_results)
            ml_reporter.report(ml_results)
