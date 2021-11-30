import argparse
import os

from utils_base_env import str_arg_to_bool, add_mysql_arguments
from utils import run_benchmarks


def main():
    benchmarks = {
        "ny_taxi": "taxi",
        "santander": "santander",
        "census": "census",
        "plasticc": "plasticc",
        "mortgage": "mortgage",
        "h2o": "h2o",
    }

    parser = argparse.ArgumentParser(description="Run benchmarks for Modin perf testing")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-bench_name",
        dest="bench_name",
        choices=sorted(benchmarks.keys()),
        help="Benchmark name.",
        required=True,
    )
    required.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded.", required=True
    )
    optional.add_argument(
        "-dfiles_num",
        dest="dfiles_num",
        default=None,
        type=int,
        help="Number of datafiles to input into database for processing.",
    )
    optional.add_argument(
        "-iterations",
        dest="iterations",
        default=1,
        type=int,
        help="Number of iterations to run every query. Best result is selected.",
    )
    optional.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default=None,
        help="Which optimizer is used",
    )
    optional.add_argument(
        "-pandas_mode",
        choices=["Pandas", "Modin_on_ray", "Modin_on_dask", "Modin_on_python", "Modin_on_omnisci"],
        default="Pandas",
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask or on Omnisci",
    )
    optional.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. It should have enough space to keep -ray_memory",
    )
    optional.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        type=int,
        help="Size of memory to allocate for Ray plasma store",
    )
    optional.add_argument(
        "-no_ml",
        default=None,
        type=str_arg_to_bool,
        help="Do not run machine learning benchmark, only ETL part",
    )
    optional.add_argument(
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
    optional.add_argument(
        "-extended_functionality",
        dest="extended_functionality",
        default=False,
        type=str_arg_to_bool,
        help="Extends functionality of H2O benchmark by adding 'chk' functions and verbose local reporting of results",
    )

    # MySQL database parameters
    add_mysql_arguments(optional, etl_ml_tables=True)
    # Omnisci server parameters
    optional.add_argument(
        "-executable", dest="executable", help="Path to omnisci_server executable."
    )

    # Additional information
    optional.add_argument(
        "-commit_omnisci",
        dest="commit_omnisci",
        default="1234567890123456789012345678901234567890",
        help="Omnisci commit hash used for benchmark.",
    )
    optional.add_argument(
        "-commit_omniscripts",
        dest="commit_omniscripts",
        default="1234567890123456789012345678901234567890",
        help="Omniscripts commit hash used for benchmark.",
    )
    optional.add_argument(
        "-commit_modin",
        dest="commit_modin",
        default="1234567890123456789012345678901234567890",
        help="Modin commit hash used for benchmark.",
    )

    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    args = parser.parse_args()

    run_benchmarks(
        benchmarks[args.bench_name],
        args.data_file,
        args.dfiles_num,
        args.iterations,
        args.validation,
        args.optimizer,
        args.pandas_mode,
        args.ray_tmpdir,
        args.ray_memory,
        args.no_ml,
        args.use_modin_xgb,
        args.gpu_memory,
        args.extended_functionality,
        args.db_server,
        args.db_port,
        args.db_user,
        args.db_pass,
        args.db_name,
        args.db_table_etl,
        args.db_table_ml,
        args.executable,
        args.commit_omnisci,
        args.commit_omniscripts,
        args.commit_modin,
    )


if __name__ == "__main__":
    main()
