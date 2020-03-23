import argparse
import subprocess
import re
import hiyapyco
import os

def str_arg_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Cannot recognize boolean value.')


def combinate_requirements(ibis, ci, res):
    merged_yaml = hiyapyco.load([ibis, ci], method=hiyapyco.METHOD_MERGE)
    with open(res, "w") as f_res:
        hiyapyco.dump(merged_yaml, stream=f_res)


def execute_process(cmdline, cwd=None, shell=False, daemon=False, print_output=True):
    "Execute cmdline in user-defined directory by creating separated process"
    try:
        print("CMD: ", " ".join(cmdline))
        output = ""
        process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=shell)
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


def convertTypeIbis2Pandas(types):
    types = ['string_' if (x == 'string') else x for x in types]
    return types

def import_pandas_into_module_namespace(namespace, mode, ray_tmpdir, ray_memory):
    if mode == 'pandas':
        print("Running on Pandas")
        import pandas as pd
    else:
        if mode == 'modin_on_ray':
            import ray
            if ray_tmpdir is None:
                ray_tmpdir = "/tmp"
            if ray_memory is None:
                ray_memory = 200*1024*1024*1024
            ray.init(huge_pages=False, plasma_directory=ray_tmpdir, memory=ray_memory, object_store_memory=ray_memory)
            os.environ["MODIN_ENGINE"] = "ray"
            print("Running on Ray on Pandas with tmp directory", ray_tmpdir, "and memory", ray_memory)
        elif mode == 'modin_on_dask':
            os.environ["MODIN_ENGINE"] = "dask"
            print("Running on Dask")
        else:
            raise ValueError(f"Unknown pandas mode {mode}")
        import modin.pandas as pd
    namespace['pd'] = pd


def equal_dfs(ibis_dfs, pandas_dfs):
    for ibis_df, pandas_df in zip(ibis_dfs, pandas_dfs):
        if not ibis_df.equals(pandas_df):
            return False
    return True

def compare_dataframes(ibis_dfs, pandas_dfs):
    prepared_dfs = []
    # in percentage - 0.05 %
    max_error = 0.05

    # preparing step
    for idx, df in enumerate(ibis_dfs):
        prepared_dfs.append(df.sort_values(by="id", axis=0).reset_index(drop=True).drop(["id"], axis=1))

    # fast check
    if equal_dfs(ibis_dfs, pandas_dfs):
        print("dataframes are equal")
        return

    # comparing step
    for ibis_df, pandas_df in zip(prepared_dfs, pandas_dfs):
        assert ibis_df.shape == pandas_df.shape
        for column_name in ibis_df.columns:
            try:
                pd.testing.assert_frame_equal(
                    ibis_df[[column_name]],
                    pandas_df[[column_name]],
                    check_less_precise=2,
                    check_dtype=False,
                )
            except AssertionError as assert_err:
                if str(ibis_df.dtypes[column_name]).startswith('float'):
                    try:
                        current_error = get_percentage(str(assert_err))
                        if current_error > max_error:
                            print(f"Max acceptable difference: {max_error}%; current difference: {current_error}%")
                            raise assert_err
                    # for catch exceptions from `get_percentage`
                    except Exception:
                        raise assert_err
                else:
                    raise assert_err

    print("dataframes are equal")
