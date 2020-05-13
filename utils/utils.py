import argparse
import glob
import os
import warnings
import re
import socket
import subprocess
from timeit import default_timer as timer
from collections import OrderedDict

import hiyapyco

returned_port_numbers = []
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


def str_arg_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Cannot recognize boolean value.")


def combinate_requirements(ibis, ci, res):
    merged_yaml = hiyapyco.load([ibis, ci], method=hiyapyco.METHOD_MERGE)
    with open(res, "w") as f_res:
        hiyapyco.dump(merged_yaml, stream=f_res)


def execute_process(cmdline, cwd=None, shell=False, daemon=False, print_output=True):
    "Execute cmdline in user-defined directory by creating separated process"
    try:
        print("CMD: ", " ".join(cmdline))
        output = ""
        process = subprocess.Popen(
            cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell
        )
        if not daemon:
            output = process.communicate()[0].strip().decode()
            if re.findall(r"\d fail", output) or re.findall(r"[e,E]rror", output):
                process.returncode = 1
            elif print_output:
                print(output)
        if process.returncode != 0 and process.returncode is not None:
            raise Exception(f"Command returned {process.returncode}. \n{output}")
        return process, output
    except OSError as err:
        print("Failed to start", cmdline, err)


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


def compare_dataframes(ibis_dfs, pandas_dfs, sort_cols=["id"], drop_cols=["id"]):
    import pandas as pd

    # in percentage - 0.05 %
    max_error = 0.05

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

    # comparing step
    for ibis_df, pandas_df in zip(ibis_dfs, pandas_dfs):
        assert ibis_df.shape == pandas_df.shape
        for column_name, column_type in ibis_df.dtypes.items():
            try:
                pd.testing.assert_frame_equal(
                    ibis_df[[column_name]],
                    pandas_df[[column_name]],
                    check_less_precise=2,
                    check_dtype=False,
                    check_categorical=False,
                )
                if str(column_type) == "category":
                    left = ibis_df[column_name]
                    right = pandas_df[column_name]
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
                if str(ibis_df.dtypes[column_name]).startswith("float"):
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


def print_results(results, backend=None, unit=""):
    results_converted = convert_units(results, ignore_fields=[], unit=unit)
    if backend:
        print(f"{backend} results:")
    for result_name, result in results_converted.items():
        print("    {} = {:.2f} {}".format(result_name, result, unit))


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def check_port_availability(port_num):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port_num))
    except Exception:
        return False
    finally:
        sock.close()
    return True


def find_free_port():
    min_port_num = 49152
    max_port_num = 65535
    if len(returned_port_numbers) == 0:
        port_num = min_port_num
    else:
        port_num = returned_port_numbers[-1] + 1
    while port_num < max_port_num:
        if check_port_availability(port_num) and port_num not in returned_port_numbers:
            returned_port_numbers.append(port_num)
            return port_num
        port_num += 1
    raise Exception("Can't find available ports")


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


class KeyValueListParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            kwargs[k] = v
        setattr(namespace, self.dest, kwargs)


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


def get_dir(dir_id):
    try:
        return directories[dir_id]
    except KeyError:
        raise ValueError(f"{dir_id} is not known")


def get_ny_taxi_dataset_size(dfiles_num):
    return sum(list(ny_taxi_data_files_sizes_MB.values())[:dfiles_num])
