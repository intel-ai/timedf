# coding: utf-8
import argparse
import os
import time

from utils_base_env import find_free_port, KeyValueListParser, str_arg_to_bool, add_mysql_arguments
from utils import remove_fields_from_dict, refactor_results_for_reporting


def main():
    args = None
    omnisci_server = None
    port_default_value = -1

    benchmarks = {
        "ny_taxi": "taxi",
        "santander": "santander",
        "census": "census",
        "plasticc": "plasticc",
        "mortgage": "mortgage",
        "h2o": "h2o",
    }
    benchmarks_list = ["ny_taxi", "santander", "census", "plasticc", "mortgage"]

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
        "-dnd", default=False, type=str_arg_to_bool, help="Do not delete old table."
    )
    optional.add_argument(
        "-dni",
        default=False,
        type=str_arg_to_bool,
        help="Do not create new table and import any data from CSV files.",
    )
    optional.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-import_mode",
        dest="import_mode",
        default="fsi",
        help="measure 'COPY FROM' import, FSI import, import through pandas",
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
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask",
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
    optional.add_argument(
        "-omnisci_cwd",
        dest="omnisci_cwd",
        help="Path to omnisci working directory. "
        "By default parent directory of executable location is used. "
        "Data directory is used in this location.",
    )
    optional.add_argument(
        "-port",
        dest="port",
        default=port_default_value,
        type=int,
        help="TCP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-http_port",
        dest="http_port",
        default=port_default_value,
        type=int,
        help="HTTP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-calcite_port",
        dest="calcite_port",
        default=port_default_value,
        type=int,
        help="Calcite port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-user", dest="user", default="admin", help="User name to use on omniscidb server."
    )
    optional.add_argument(
        "-password",
        dest="password",
        default="HyperInteractive",
        help="User password to use on omniscidb server.",
    )
    optional.add_argument(
        "-database_name",
        dest="database_name",
        default="omnisci",
        help="Database name to use in omniscidb server.",
    )
    optional.add_argument(
        "-table",
        dest="table",
        default="benchmark_table",
        help="Table name name to use in omniscidb server.",
    )
    optional.add_argument(
        "-ipc_conn",
        dest="ipc_conn",
        default=True,
        type=str_arg_to_bool,
        help="Table name name to use in omniscidb server.",
    )
    optional.add_argument(
        "-debug_timer",
        dest="debug_timer",
        default=False,
        type=str_arg_to_bool,
        help="Enable fine-grained query execution timers for debug.",
    )
    optional.add_argument(
        "-columnar_output",
        dest="columnar_output",
        default=True,
        type=str_arg_to_bool,
        help="Allows OmniSci Core to directly materialize intermediate projections \
            and the final ResultSet in Columnar format where appropriate.",
    )
    optional.add_argument(
        "-lazy_fetch",
        dest="lazy_fetch",
        default=None,
        type=str_arg_to_bool,
        help="[lazy_fetch help message]",
    )
    optional.add_argument(
        "-multifrag_rs",
        dest="multifrag_rs",
        default=None,
        type=str_arg_to_bool,
        help="[multifrag_rs help message]",
    )
    optional.add_argument(
        "-fragments_size",
        dest="fragments_size",
        default=None,
        nargs="*",
        type=int,
        help="Number of rows per fragment that is a unit of the table for query processing. \
            Should be specified for each table in workload",
    )
    optional.add_argument(
        "-omnisci_run_kwargs",
        dest="omnisci_run_kwargs",
        default={},
        metavar="KEY1=VAL1,KEY2=VAL2...",
        action=KeyValueListParser,
        help="options to start omnisci server",
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
    optional.add_argument(
        "-debug_mode",
        dest="debug_mode",
        default=False,
        type=str_arg_to_bool,
        help="Enable debug mode.",
    )

    try:
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUNBUFFERED"] = "1"
        omnisci_server_worker = None
        omnisci_server = None

        args = parser.parse_args()

        launch_omnisci_server = args.bench_name in benchmarks_list and args.pandas_mode == "Modin_on_omnisci"

        if args.port == port_default_value:
            args.port = find_free_port()
        if args.http_port == port_default_value:
            args.http_port = find_free_port()
        if args.calcite_port == port_default_value:
            args.calcite_port = find_free_port()

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
            "debug_mode": args.debug_mode,
            "extended_functionality": args.extended_functionality,
        }

        if launch_omnisci_server:
            if args.executable is None:
                parser.error(
                    "Omnisci executable should be specified with -e/--executable"
                )
            from server import OmnisciServer

            omnisci_server = OmnisciServer(
                omnisci_executable=args.executable,
                omnisci_port=args.port,
                http_port=args.http_port,
                calcite_port=args.calcite_port,
                database_name=args.database_name,
                omnisci_cwd=args.omnisci_cwd,
                user=args.user,
                password=args.password,
                debug_timer=args.debug_timer,
                columnar_output=args.columnar_output,
                lazy_fetch=args.lazy_fetch,
                multifrag_rs=args.multifrag_rs,
                omnisci_run_kwargs=args.omnisci_run_kwargs,
            )

            parameters["database_name"] = args.database_name
            parameters["table"] = args.table
            parameters["dnd"] = args.dnd
            parameters["dni"] = args.dni
            parameters["import_mode"] = args.import_mode
            parameters["fragments_size"] = args.fragments_size

        etl_results = []
        ml_results = []
        print(parameters)
        run_id = int(round(time.time()))
        for iter_num in range(1, args.iterations + 1):
            print(f"Iteration #{iter_num}")

            if launch_omnisci_server:
                from server_worker import OmnisciServerWorker

                omnisci_server_worker = OmnisciServerWorker(omnisci_server)
                parameters["omnisci_server_worker"] = omnisci_server_worker
                parameters["ipc_connection"] = args.ipc_conn
                omnisci_server.launch()
            parameters = {
                key: os.path.expandvars(value) if isinstance(value, str) else value
                for key, value in parameters.items()
            }
            benchmark_results = run_benchmark(parameters)

            if launch_omnisci_server:
                omnisci_server_worker.terminate()
                omnisci_server.terminate()

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

    finally:
        if omnisci_server_worker:
            omnisci_server_worker.terminate()
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == "__main__":
    main()
