import argparse
import os
import socket
import subprocess
from typing import Union

returned_port_numbers = []


def str_arg_to_bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Cannot recognize boolean value.")


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
        # No `None` value indicates that the process has terminated
        if process.returncode is not None:
            if process.returncode != 0:
                raise Exception(f"{output}\n\nCommand returned {process.returncode}.")
            if print_output:
                print(output)
        return process, output
    except OSError as err:
        print("Failed to start", cmdline, err)


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


class KeyValueListParser(argparse.Action):
    def __call__(self, parser, namespace, values: str, option_string=None):
        kwargs = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            kwargs[k] = v
        setattr(namespace, self.dest, kwargs)


def add_mysql_arguments(parser, etl_ml_tables=False):
    parser.add_argument(
        "-db_server", dest="db_server", default="localhost", help="Host name of MySQL server."
    )
    parser.add_argument(
        "-db_port", dest="db_port", default=3306, type=int, help="Port number of MySQL server."
    )
    parser.add_argument(
        "-db_user",
        dest="db_user",
        help="Username to use to connect to MySQL database. "
        "If user name is specified, script attempts to store results in MySQL "
        "database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db_pass",
        dest="db_pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database.",
    )
    parser.add_argument(
        "-db_name",
        dest="db_name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results.",
    )
    if etl_ml_tables:
        parser.add_argument(
            "-db_table_etl",
            dest="db_table_etl",
            help="Table to store ETL results for this benchmark.",
        )
        parser.add_argument(
            "-db_table_ml",
            dest="db_table_ml",
            help="Table to store ML results for this benchmark.",
        )
    else:
        parser.add_argument(
            "-db-table", dest="db_table", help="Table to use to store results for this benchmark."
        )


def prepare_parser():
    # be careful with this line when moving `prepare_parser` function
    omniscript_path = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="Run benchmarks for Modin perf testing")
    required = parser.add_argument_group("common")
    optional = parser.add_argument_group("optional arguments")
    hdk = parser.add_argument_group("hdk")
    benchmark = parser.add_argument_group("benchmark")
    mysql = parser.add_argument_group("mysql")
    commits = parser.add_argument_group("commits")

    possible_tasks = ["benchmark"]
    benchmarks = ["ny_taxi", "santander", "census", "plasticc", "mortgage", "h2o", "taxi_ml"]

    # Task
    required.add_argument(
        "-task",
        dest="task",
        required=True,
        help=f"Task for execute {possible_tasks}. Use , separator for multiple tasks",
    )

    # Environment
    optional.add_argument(
        "-en", "--env_name", dest="env_name", default=None, help="Conda env name."
    )
    optional.add_argument(
        "-ec",
        "--env_check",
        dest="env_check",
        default=False,
        type=str_arg_to_bool,
        help="Check if env exists. If it exists don't recreate. Ignored if `--env_name` isn't set.",
    )
    optional.add_argument(
        "-s",
        "--save_env",
        dest="save_env",
        default=False,
        type=str_arg_to_bool,
        help="Save conda env after executing. Ignored if `--env_name` isn't set.",
    )
    optional.add_argument(
        "-ci",
        "--ci_requirements",
        dest="ci_requirements",
        default=os.path.join(omniscript_path, "ci_requirements.yml"),
        help="File with ci requirements for conda env.",
    )
    optional.add_argument(
        "-py",
        "--python_version",
        dest="python_version",
        default="3.8",
        help="Python version that should be installed in conda env.",
    )
    # Modin
    optional.add_argument(
        "-m", "--modin_path", dest="modin_path", default=None, help="Path to modin directory."
    )
    optional.add_argument(
        "--modin_pkgs_dir",
        dest="modin_pkgs_dir",
        default=None,
        type=str,
        help="Path where to store built Modin dependencies (--target flag for pip), can be helpful if you have space limited home directory.",
    )

    # Benchmark parameters
    benchmark.add_argument(
        "-bench_name", dest="bench_name", choices=benchmarks, help="Benchmark name."
    )
    benchmark.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded."
    )
    benchmark.add_argument(
        "-dfiles_num",
        dest="dfiles_num",
        default=None,
        type=int,
        help="Number of datafiles to load into database for processing.",
    )
    benchmark.add_argument(
        "-iterations",
        dest="iterations",
        default=1,
        type=int,
        help="Number of iterations to run. All results will be submitted to the DB.",
    )
    benchmark.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    benchmark.add_argument(
        "-optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default=None,
        help="Optimizer to use.",
    )
    benchmark.add_argument(
        "-pandas_mode",
        choices=["Pandas", "Modin_on_ray", "Modin_on_dask", "Modin_on_python", "Modin_on_hdk"],
        default="Pandas",
        help="Specifies which version of Pandas to use: "
        "plain Pandas, Modin runing on Ray or on Dask or on HDK",
    )
    benchmark.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. "
        "It should have enough space to keep -ray_memory",
    )
    benchmark.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        type=int,
        help="Size of memory to allocate for Ray plasma store",
    )
    benchmark.add_argument(
        "-no_ml",
        default=None,
        type=str_arg_to_bool,
        help="Do not run machine learning benchmark, only ETL part",
    )
    benchmark.add_argument(
        "-use_modin_xgb",
        default=False,
        type=str_arg_to_bool,
        help="Whether to use Modin XGBoost for ML part, relevant for Plasticc benchmark only",
    )
    optional.add_argument(
        "-gpu_memory",
        dest="gpu_memory",
        type=int,
        help="specify the memory of your gpu"
        "(This controls the lines to be used. Also work for CPU version. )",
        default=None,
    )
    benchmark.add_argument(
        "-extended_functionality",
        dest="extended_functionality",
        default=False,
        type=str_arg_to_bool,
        help="Extends functionality of H2O benchmark by adding 'chk' functions and verbose local reporting of results",
    )
    # MySQL database parameters
    add_mysql_arguments(mysql, etl_ml_tables=True)
    # Additional information
    commits.add_argument(
        "-commit_hdk",
        dest="commit_hdk",
        default="1234567890123456789012345678901234567890",
        help="HDK commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_omniscripts",
        dest="commit_omniscripts",
        default="1234567890123456789012345678901234567890",
        help="Omniscripts commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_modin",
        dest="commit_modin",
        default="1234567890123456789012345678901234567890",
        help="Modin commit hash used for tests.",
    )
    return parser, possible_tasks, omniscript_path
