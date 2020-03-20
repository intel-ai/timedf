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
