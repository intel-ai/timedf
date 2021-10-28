# coding: utf-8
import argparse
import os
import time

from utils_base_env import str_arg_to_bool, add_mysql_arguments
from utils import remove_fields_from_dict, refactor_results_for_reporting


def main():
    args = None

    benchmarks = {
        "ny_taxi": "taxi",
        "santander": "santander",
        "census": "census",
        "plasticc": "plasticc",
        "mortgage": "mortgage",
        "h2o": "h2o",
    }

    ignore_fields_for_bd_report_etl = ["t_connect"]
    ignore_fields_for_bd_report_ml = []
    ignore_fields_for_results_unit_conversion = [
        "Backend",
        "dfiles_num",
        "dataset_size",
        "query_name",
    ]

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

    run_benchmark = __import__(benchmarks[args.bench_name]).run_benchmark

    parameters = {
        "data_file": args.data_file,
        "dfiles_num": args.dfiles_num,
        "no_ml": args.no_ml,
        "use_modin_xgb": args.use_modin_xgb,
        "optimizer": args.optimizer,
        "pandas_mode": args.pandas_mode,
        "ray_tmpdir": args.ray_tmpdir,
        "ray_memory": args.ray_memory,
        "gpu_memory": args.gpu_memory,
        "validation": args.validation,
        "extended_functionality": args.extended_functionality,
    }

    etl_results = []
    ml_results = []
    print(parameters)
    run_id = int(round(time.time()))
    for iter_num in range(1, args.iterations + 1):
        print(f"Iteration #{iter_num}")

        parameters = {
            key: os.path.expandvars(value) if isinstance(value, str) else value
            for key, value in parameters.items()
        }
        benchmark_results = run_benchmark(parameters)

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

        # Reporting to MySQL database
        if args.db_user is not None:
            import mysql.connector
            from report import DbReport

            if iter_num == 1:
                db = mysql.connector.connect(
                    host=args.db_server,
                    port=args.db_port,
                    user=args.db_user,
                    passwd=args.db_pass,
                    db=args.db_name,
                )

                reporting_init_fields = {
                    "OmnisciCommitHash": args.commit_omnisci,
                    "OmniscriptsCommitHash": args.commit_omniscripts,
                    "ModinCommitHash": args.commit_modin,
                }

                reporting_fields_benchmark_etl = {
                    x: "VARCHAR(500) NOT NULL" for x in etl_results[0]
                }
                if len(etl_results) != 1:
                    reporting_fields_benchmark_etl.update(
                        {x: "VARCHAR(500) NOT NULL" for x in etl_results[1]}
                    )

                db_reporter_etl = DbReport(
                    db,
                    args.db_table_etl,
                    reporting_fields_benchmark_etl,
                    reporting_init_fields,
                )

                if len(ml_results) != 0:
                    reporting_fields_benchmark_ml = {
                        x: "VARCHAR(500) NOT NULL" for x in ml_results[0]
                    }
                    if len(ml_results) != 1:
                        reporting_fields_benchmark_ml.update(
                            {x: "VARCHAR(500) NOT NULL" for x in ml_results[1]}
                        )

                    db_reporter_ml = DbReport(
                        db,
                        args.db_table_ml,
                        reporting_fields_benchmark_ml,
                        reporting_init_fields,
                    )

            if iter_num == args.iterations:
                for result_etl in etl_results:
                    remove_fields_from_dict(result_etl, ignore_fields_for_bd_report_etl)
                    db_reporter_etl.submit(result_etl)

                if len(ml_results) != 0:
                    for result_ml in ml_results:
                        remove_fields_from_dict(result_ml, ignore_fields_for_bd_report_ml)
                        db_reporter_ml.submit(result_ml)


if __name__ == "__main__":
    main()
