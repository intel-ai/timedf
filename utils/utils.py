import glob
import os
import warnings
from timeit import default_timer as timer
from collections import OrderedDict
import psutil
from tempfile import mkstemp

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


def convert_type_ibis2pandas(types):
    types = ["string_" if (x == "string") else x for x in types]
    return types


def import_pandas_into_module_namespace(namespace, mode, ray_tmpdir=None, ray_memory=None):
    if mode == "Pandas":
        print("Pandas backend: pure Pandas")
        import pandas as pd
    else:
        if mode == "Modin_on_ray":
            import ray

            if not ray_tmpdir:
                ray_tmpdir = "/tmp"
            if not ray_memory:
                ray_memory = 200 * 1024 * 1024 * 1024
            if not ray.is_initialized():
                ray.init(
                    huge_pages=False,
                    plasma_directory=ray_tmpdir,
                    memory=ray_memory,
                    object_store_memory=ray_memory,
                )
            os.environ["MODIN_ENGINE"] = "ray"
            print(
                f"Pandas backend: Modin on Ray with tmp directory {ray_tmpdir} and memory {ray_memory}"
            )
        elif mode == "Modin_on_dask":
            os.environ["MODIN_ENGINE"] = "dask"
            print("Pandas backend: Modin on Dask")
        elif mode == "Modin_on_python":
            os.environ["MODIN_ENGINE"] = "python"
            print("Pandas backend: Modin on pure Python")
        elif mode == "Modin_on_omnisci":
            os.environ["MODIN_ENGINE"] = "ray"
            os.environ["MODIN_BACKEND"] = "omnisci"
            os.environ["MODIN_EXPERIMENTAL"] = "True"
            print("Pandas backend: Modin on OmniSci")
        else:
            raise ValueError(f"Unknown pandas mode {mode}")
        import modin.pandas as pd
    if not isinstance(namespace, (list, tuple)):
        namespace = [namespace]
    for space in namespace:
        space["pd"] = pd


def equal_dfs(ibis_dfs, pandas_dfs):
    for ibis_df, pandas_df in zip(ibis_dfs, pandas_dfs):
        if not ibis_df.equals(pandas_df):
            return False
    return True


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


def compare_dataframes(
    ibis_dfs, pandas_dfs, sort_cols=["id"], drop_cols=["id"], parallel_execution=False
):
    import pandas as pd

    parallel_processes = os.cpu_count() // 2

    t0 = timer()
    assert len(ibis_dfs) == len(pandas_dfs)

    # preparing step
    for idx in range(len(ibis_dfs)):
        # prepare ibis part
        if isinstance(ibis_dfs[idx], pd.Series):
            # that means, that indexes in Series must be the same
            # as 'id' column in source dataframe
            ibis_dfs[idx].sort_index(axis=0, inplace=True)
        else:
            if sort_cols:
                ibis_dfs[idx].sort_values(by=sort_cols, axis=0, inplace=True)
            if drop_cols:
                ibis_dfs[idx].drop(drop_cols, axis=1, inplace=True)

        ibis_dfs[idx].reset_index(drop=True, inplace=True)
        # prepare pandas part
        pandas_dfs[idx].reset_index(drop=True, inplace=True)

    # fast check
    if equal_dfs(ibis_dfs, pandas_dfs):
        print("dataframes are equal")
        return

    print("Fast check took {:.2f} seconds".format(timer() - t0))

    # comparing step
    t0 = timer()
    for ibis_df, pandas_df in zip(ibis_dfs, pandas_dfs):
        assert ibis_df.shape == pandas_df.shape
        if parallel_execution:
            from multiprocessing import Pool

            pool = Pool(parallel_processes)
            pool.map(
                compare_columns,
                (
                    (ibis_df[column_name], pandas_df[column_name])
                    for column_name in ibis_df.columns
                ),
            )
            pool.close()
        else:
            for column_name in ibis_df.columns:
                compare_columns((ibis_df[column_name], pandas_df[column_name]))

        print("Per-column check took {:.2f} seconds".format(timer() - t0))

    print("dataframes are equal")


def load_data_pandas(
    filename,
    columns_names=None,
    columns_types=None,
    header=None,
    nrows=None,
    use_gzip=False,
    parse_dates=None,
    pd=None,
):
    if not pd:
        import_pandas_into_module_namespace(namespace=load_data_pandas.__globals__, mode="Pandas")
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


def files_names_from_pattern(filename):
    from braceexpand import braceexpand

    data_files_names = list(braceexpand(filename))
    data_files_names = sorted([x for f in data_files_names for x in glob.glob(f)])
    return data_files_names


def print_times(times, backend=None):
    if backend:
        print(f"{backend} times:")
    for time_name, time in times.items():
        print("{} = {:.5f} s".format(time_name, time))


def print_results(results, backend=None, unit="", ignore_fields=[]):
    results_converted = convert_units(results, ignore_fields=[], unit=unit)
    if backend:
        print(f"{backend} results:")
    for result_name, result in results_converted.items():
        if result_name not in ignore_fields:
            print("    {} = {:.5f} {}".format(result_name, result, unit))


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def split(X, y, test_size=0.1, stratify=None, random_state=None, optimizer="intel"):
    if optimizer == "intel":
        import daal4py  # noqa: F401 (imported but unused) FIXME
        from daal4py import sklearn  # noqa: F401 (imported but unused) FIXME
        from sklearn.model_selection import train_test_split
    elif optimizer == "stock":
        from sklearn.model_selection import train_test_split
    else:
        raise ValueError(
            f"Intel optimized and stock sklearn are supported. \
            {optimizer} can't be recognized"
        )

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
        if key in dictonary:
            dictonary.pop(key)


def convert_units(dict_to_convert, ignore_fields, unit="ms"):
    try:
        multiplier = conversions[unit]
    except KeyError:
        raise ValueError(f"Conversion to {unit} is not implemented")

    return {
        key: (value * multiplier if key not in ignore_fields else value)
        for key, value in dict_to_convert.items()
    }


def check_fragments_size(fragments_size, count_table, import_mode, default_fragments_size=None):
    result_fragments_size = []
    check_options = {
        "fragments_size": fragments_size,
        "default_fragments_size": default_fragments_size,
    }

    for option_name, option in check_options.items():
        if option:
            if import_mode != "pandas" and len(option) != count_table:
                raise ValueError(
                    f"fragment size should be specified for each table; \
                    {option_name}: {option}; count table: {count_table}"
                )

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
    return process.memory_info().rss / (1024 ** 3)  # GB units


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
        data_id: os.path.getsize(data_file) / 1024 / 1024
        for data_id, data_file in data_files_paths.items()
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
                    file_to_write=file_name, output_file=self._data_file_path, write_mode="ab",
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
            backend_result_converted = []
            backend_result_values_list = list(backend_result.values()) if backend_result else None
            if backend_result is not None and all(
                [
                    isinstance(backend_result_values_list[i], dict)
                    for i in range(len(backend_result_values_list))
                ]
            ):  # True if subqueries are used
                for query_name, query_results in backend_result.items():
                    query_results.update({"query_name": query_name})
                    backend_result_converted.append(query_results)
            else:
                backend_result_converted.append(backend_result)

            for result in backend_result_converted:
                if result:
                    result = convert_units(
                        result,
                        ignore_fields=ignore_fields_for_results_unit_conversion,
                        unit=reporting_unit,
                    )
                    category_additional_fields = additional_fields.get(results_category, None)
                    if category_additional_fields:
                        for field in category_additional_fields.keys():
                            result[field] = category_additional_fields[field]
                    etl_ml_results[results_category].append(result)

    return etl_ml_results
