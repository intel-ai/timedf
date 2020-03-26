import argparse
import glob
import os
import sys
import time

import pandas as pd

import mysql.connector
from braceexpand import braceexpand

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from report import DbReport
from server import OmnisciServer
from server_worker import OmnisciServerWorker


def compare_tables(table1, table2):

    if table1.equals(table2):
        return True
    else:
        print("\ntables are not equal, table1:")

        print(type(table1))
        print(table1)
        print("\ntable2:")

        print(type(table2))
        print(table2)
        return False


def validation_prereqs():
    return omnisci_server_worker.import_data_by_pandas(
        data_files_names=data_files_names,
        files_limit=args.df,
        columns_names=taxibench_columns_names,
    )


# Queries definitions
def q1():
    t_query = 0
    t0 = time.time()
    q1_output_ibis = (
        df.groupby("cab_type")
        .count()
        .sort_by("cab_type")["cab_type", "count"]
        .execute()
    )
    t_query += time.time() - t0

    if args.val and not queries_validation_flags["q1"]:
        print("Validating query 1 results ...")

        queries_validation_flags["q1"] = True

        q1_output_pd = df_pandas.groupby("cab_type")["cab_type"].count()

        # Casting of Pandas q1 output to Pandas.DataFrame type, which is compartible with
        # Ibis q1 output
        q1_output_pd_df = q1_output_pd.to_frame()
        q1_output_pd_df.loc[:, "count"] = q1_output_pd_df.loc[:, "cab_type"]
        q1_output_pd_df["cab_type"] = q1_output_pd_df.index
        q1_output_pd_df.index = [i for i in range(len(q1_output_pd_df))]

        queries_validation_results["q1"] = compare_tables(
            q1_output_pd_df, q1_output_ibis
        )
        if queries_validation_results["q1"]:
            print("q1 results are validated!")

    return t_query


def q2():
    t_query = 0
    t0 = time.time()
    q2_output_ibis = (
        df.groupby("passenger_count")
        .aggregate(total_amount=df.total_amount.mean())[
            ["passenger_count", "total_amount"]
        ]
        .execute()
    )
    t_query += time.time() - t0

    if args.val and not queries_validation_flags["q2"]:
        print("Validating query 2 results ...")

        queries_validation_flags["q2"] = True

        q2_output_pd = df_pandas.groupby("passenger_count", as_index=False).mean()[
            ["passenger_count", "total_amount"]
        ]

        queries_validation_results["q2"] = compare_tables(q2_output_pd, q2_output_ibis)
        if queries_validation_results["q2"]:
            print("q2 results are validated!")

    return t_query


def q3():
    t_query = 0
    t0 = time.time()
    q3_output_ibis = (
        df.groupby(
            [df.passenger_count, df.pickup_datetime.year().name("pickup_datetime")]
        )
        .aggregate(count=df.passenger_count.count())
        .execute()
    )
    t_query += time.time() - t0

    if args.val and not queries_validation_flags["q3"]:
        print("Validating query 3 results ...")

        queries_validation_flags["q3"] = True

        transformed = df_pandas[["passenger_count", "pickup_datetime"]].transform(
            {
                "passenger_count": lambda x: x,
                "pickup_datetime": lambda x: pd.DatetimeIndex(x).year,
            }
        )
        q3_output_pd = transformed.groupby(["passenger_count", "pickup_datetime"])[
            ["passenger_count", "pickup_datetime"]
        ].count()["passenger_count"]

        # Casting of Pandas q3 output to Pandas.DataFrame type, which is compartible with
        # Ibis q3 output
        q3_output_pd_df = q3_output_pd.to_frame()
        count_df = q3_output_pd_df.loc[:, "passenger_count"].copy()
        q3_output_pd_df["passenger_count"] = q3_output_pd.index.droplevel(
            level="pickup_datetime"
        )
        q3_output_pd_df["pickup_datetime"] = q3_output_pd.index.droplevel(
            level="passenger_count"
        )
        q3_output_pd_df = q3_output_pd_df.astype({"pickup_datetime": "int32"})
        q3_output_pd_df.loc[:, "count"] = count_df
        q3_output_pd_df.index = [i for i in range(len(q3_output_pd_df))]

        queries_validation_results["q3"] = compare_tables(
            q3_output_pd_df, q3_output_ibis
        )
        if queries_validation_results["q3"]:
            print("q3 results are validated!")

    return t_query


def q4():
    t_query = 0
    t0 = time.time()
    q4_ibis_sized = df.groupby(
        [
            df.passenger_count,
            df.pickup_datetime.year().name("pickup_datetime"),
            df.trip_distance.cast("int64").name("trip_distance"),
        ]
    ).size()
    q4_output_ibis = q4_ibis_sized.sort_by(
        [("pickup_datetime", True), ("count", False)]
    ).execute()
    t_query += time.time() - t0

    if args.val and not queries_validation_flags["q4"]:
        print("Validating query 4 results ...")

        queries_validation_flags["q4"] = True

        q4_pd_sized = (
            df_pandas[["passenger_count", "pickup_datetime", "trip_distance"]]
            .transform(
                {
                    "passenger_count": lambda x: x,
                    "pickup_datetime": lambda x: pd.DatetimeIndex(x).year,
                    "trip_distance": lambda x: x.astype("int64", copy=False),
                }
            )
            .groupby(["passenger_count", "pickup_datetime", "trip_distance"])
            .size()
            .reset_index()
        )

        q4_output_pd = q4_pd_sized.sort_values(
            by=["pickup_datetime", 0], ascending=[True, False]
        )

        # Casting of Pandas q4 output to Pandas.DataFrame type, which is compartible with
        # Ibis q4 output
        q4_output_pd = q4_output_pd.astype({"pickup_datetime": "int32"})
        q4_output_pd.columns = [
            "passenger_count",
            "pickup_datetime",
            "trip_distance",
            "count",
        ]
        q4_output_pd.index = [i for i in range(len(q4_output_pd))]

        # compare_result_1 and compare_result_2 are the results of comparison of q4 sorted columns
        compare_result_1 = compare_tables(
            q4_output_pd["pickup_datetime"], q4_output_ibis["pickup_datetime"]
        )
        compare_result_2 = compare_tables(
            q4_output_pd["count"], q4_output_ibis["count"]
        )

        # compare_result_3 is the result of q4 output table all elements presence check
        q4_output_ibis_validation = q4_ibis_sized.sort_by(
            [
                ("pickup_datetime", True),
                ("count", False),
                ("trip_distance", True),
                ("passenger_count", True),
            ]
        ).execute()
        q4_output_pd_valid = q4_pd_sized.sort_values(
            by=["trip_distance", "passenger_count"]
        ).sort_values(by=["pickup_datetime", 0], ascending=[True, False])
        q4_output_pd_valid = q4_output_pd_valid.astype({"pickup_datetime": "int32"})
        q4_output_pd_valid.columns = [
            "passenger_count",
            "pickup_datetime",
            "trip_distance",
            "count",
        ]
        q4_output_pd_valid.index = [i for i in range(len(q4_output_pd))]

        compare_result_3 = compare_tables(q4_output_pd_valid, q4_output_ibis_validation)

        queries_validation_results["q4"] = (
            compare_result_1 and compare_result_2 and compare_result_3
        )
        if queries_validation_results["q4"]:
            print("q4 results are validated!")

    return t_query


queries_list = [q1, q2, q3, q4]
queries_description = {}
queries_description[1] = "NYC taxi query 1"
queries_description[2] = "NYC taxi query 2"
queries_description[3] = "NYC taxi query 3"
queries_description[4] = "NYC taxi query 4"


omnisci_executable = "../omnisci/build/bin/omnisci_server"
taxi_trips_directory = "/localdisk/work/trips_x*.csv"
taxibench_table_name = "trips"
tmp_table_name = "tmp_table"
omnisci_server = None
queries_validation_results = {"q%s" % i: False for i in range(1, 5)}
queries_validation_flags = {"q%s" % i: False for i in range(1, 5)}
validation_prereqs_flag = False

parser = argparse.ArgumentParser(description="Run NY Taxi benchmark using Ibis.")

parser.add_argument(
    "-e", default=omnisci_executable, help='Path to executable "omnisci_server".'
)
parser.add_argument("-r", default="report_taxibench_ibis.csv", help="Report file name.")
parser.add_argument(
    "-df",
    default=1,
    type=int,
    help="Number of datafiles to input into database for processing.",
)
parser.add_argument(
    "-dp",
    default=taxi_trips_directory,
    help="Wildcard pattern of datafiles that should be loaded.",
)
parser.add_argument(
    "-i",
    default=5,
    type=int,
    help="Number of iterations to run every query. Best result is selected.",
)
parser.add_argument("-dnd", action="store_true", help="Do not delete old table.")
parser.add_argument(
    "-dni",
    action="store_true",
    help="Do not create new table and import any data from CSV files.",
)
parser.add_argument(
    "-val",
    action="store_true",
    help="validate queries results (by comparison with Pandas queries results).",
)
parser.add_argument(
    "-port",
    default=62074,
    type=int,
    help="TCP port that omnisql client should use to connect to server.",
)
parser.add_argument("-u", default="admin", help="User name to use on omniscidb server.")
parser.add_argument(
    "-p", default="HyperInteractive", help="User password to use on omniscidb server."
)
parser.add_argument(
    "-n", default="agent_test_ibis", help="Database name to use on omniscidb server."
)
parser.add_argument(
    "-db-server", default="localhost", help="Host name of MySQL server."
)
parser.add_argument(
    "-db-port", default=3306, type=int, help="Port number of MySQL server."
)
parser.add_argument(
    "-db-user",
    default="",
    help="Username to use to connect to MySQL database. If user name is specified,\
                    script attempts to store results in MySQL database using other -db-parameters.",
)
parser.add_argument(
    "-db-pass",
    default="omniscidb",
    help="Password to use to connect to MySQL database.",
)
parser.add_argument(
    "-db-name",
    default="omniscidb",
    help="MySQL database to use to store benchmark results.",
)
parser.add_argument(
    "-db-table", help="Table to use to store results for this benchmark."
)

parser.add_argument(
    "-commit_omnisci",
    dest="commit_omnisci",
    default="1234567890123456789012345678901234567890",
    help="Omnisci commit hash to use for tests.",
)
parser.add_argument(
    "-commit_ibis",
    dest="commit_ibis",
    default="1234567890123456789012345678901234567890",
    help="Ibis commit hash to use for tests.",
)

try:
    args = parser.parse_args()
    if args.df <= 0:
        print("Bad number of data files specified", args.df)
        sys.exit(1)

    if args.i < 1:
        print("Bad number of iterations specified", args.i)

    database_name = args.n
    omnisci_server = OmnisciServer(
        omnisci_executable=args.e,
        omnisci_port=args.port,
        database_name=database_name,
        user=args.u,
        password=args.p,
    )
    omnisci_server.launch()
    omnisci_server_worker = OmnisciServerWorker(omnisci_server)

    time.sleep(2)
    omnisci_server_worker.connect_to_server()

    taxibench_columns_names = [
        "trip_id",
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "store_and_fwd_flag",
        "rate_code_id",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "trip_type",
        "pickup",
        "dropoff",
        "cab_type",
        "precipitation",
        "snow_depth",
        "snowfall",
        "max_temperature",
        "min_temperature",
        "average_wind_speed",
        "pickup_nyct2010_gid",
        "pickup_ctlabel",
        "pickup_borocode",
        "pickup_boroname",
        "pickup_ct2010",
        "pickup_boroct2010",
        "pickup_cdeligibil",
        "pickup_ntacode",
        "pickup_ntaname",
        "pickup_puma",
        "dropoff_nyct2010_gid",
        "dropoff_ctlabel",
        "dropoff_borocode",
        "dropoff_boroname",
        "dropoff_ct2010",
        "dropoff_boroct2010",
        "dropoff_cdeligibil",
        "dropoff_ntacode",
        "dropoff_ntaname",
        "dropoff_puma",
    ]

    taxibench_columns_types = [
        "int64",
        "int64",
        "timestamp",
        "timestamp",
        "string",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
    ]

    db_reporter = None
    if args.db_user is not "":
        print("Connecting to database")
        db = mysql.connector.connect(
            host=args.db_server,
            port=args.db_port,
            user=args.db_user,
            passwd=args.db_pass,
            db=args.db_name,
        )
        db_reporter = DbReport(
            db,
            args.db_table,
            {
                "FilesNumber": "INT UNSIGNED NOT NULL",
                "QueryName": "VARCHAR(500) NOT NULL",
                "FirstExecTimeMS": "BIGINT UNSIGNED",
                "WorstExecTimeMS": "BIGINT UNSIGNED",
                "BestExecTimeMS": "BIGINT UNSIGNED",
                "AverageExecTimeMS": "BIGINT UNSIGNED",
                "TotalTimeMS": "BIGINT UNSIGNED",
                "QueryValidation": "VARCHAR(500) NOT NULL",
                "IbisCommitHash": "VARCHAR(500) NOT NULL",
            },
            {"ScriptName": "taxibench_ibis.py", "CommitHash": args.commit_omnisci},
        )

    # Delete old table
    if not args.dnd:
        print("Deleting", database_name, "old database")
        try:
            omnisci_server_worker.get_conn().drop_database(database_name, force=True)
            time.sleep(2)
            omnisci_server_worker.connect_to_server()
        except Exception as err:
            print("Failed to delete", database_name, "old database: ", err)

    args.dp = args.dp.replace("'", "")
    data_files_names = list(braceexpand(args.dp))
    data_files_names = sorted([x for f in data_files_names for x in glob.glob(f)])

    data_files_number = len(data_files_names[: args.df])

    try:
        print("Creating", database_name, "new database")
        omnisci_server_worker.get_conn().create_database(
            database_name
        )  # Ibis list_databases method is not supported yet
    except Exception as err:
        print("Database creation is skipped, because of error:", err)

    if len(data_files_names) == 0:
        print("Could not find any data files matching", args.dp)
        sys.exit(2)

    # Create table and import data
    if not args.dni:
        # Datafiles import
        omnisci_server_worker.import_data(
            table_name=taxibench_table_name,
            data_files_names=data_files_names,
            files_limit=args.df,
            columns_names=taxibench_columns_names,
            columns_types=taxibench_columns_types,
            header=False,
        )

    try:
        db = omnisci_server_worker.database(database_name)
    except Exception as err:
        print("Failed to connect to database:", err)

    try:
        tables_names = db.list_tables()
        print("Database tables:", tables_names)
    except Exception as err:
        print("Failed to read database tables:", err)

    try:
        df = db.table(taxibench_table_name)
    except Exception as err:
        print("Failed to access", taxibench_table_name, "table:", err)

    try:
        if args.val:
            df_pandas = validation_prereqs()

        with open(args.r, "w") as report:
            t_begin = time.time()
            for query_number in range(0, 4):
                exec_times = [None] * 5
                best_exec_time = float("inf")
                worst_exec_time = 0.0
                first_exec_time = float("inf")
                times_sum = 0.0
                for iteration in range(1, args.i + 1):
                    print(
                        "Running query number:",
                        query_number + 1,
                        "Iteration number:",
                        iteration,
                    )
                    time_tmp = int(round(queries_list[query_number]() * 1000))
                    exec_times[iteration - 1] = time_tmp
                    if iteration == 1:
                        first_exec_time = exec_times[iteration - 1]
                    if best_exec_time > exec_times[iteration - 1]:
                        best_exec_time = exec_times[iteration - 1]
                    if iteration != 1 and worst_exec_time < exec_times[iteration - 1]:
                        worst_exec_time = exec_times[iteration - 1]
                    if iteration != 1:
                        times_sum += exec_times[iteration - 1]
                average_exec_time = times_sum / (args.i - 1)
                total_exec_time = int(round(time.time() - t_begin))
                print(
                    "Query",
                    query_number + 1,
                    "Exec time (ms):",
                    best_exec_time,
                    "Total time (s):",
                    total_exec_time,
                )
                print(
                    "FilesNumber: ",
                    str(data_files_number),
                    ",",
                    "QueryName: ",
                    queries_description[query_number + 1],
                    ",",
                    "IbisCommitHash",
                    args.commit_ibis,
                    ",",
                    "FirstExecTimeMS: ",
                    first_exec_time,
                    ",",
                    "WorstExecTimeMS: ",
                    worst_exec_time,
                    ",",
                    "BestExecTimeMS: ",
                    best_exec_time,
                    ",",
                    "AverageExecTimeMS: ",
                    average_exec_time,
                    ",",
                    "QueryValidation: ",
                    str(queries_validation_results["q%s" % (query_number + 1)]),
                    ",",
                    "TotalTimeMS: ",
                    total_exec_time,
                    ",",
                    "",
                    "\n",
                    file=report,
                    sep="",
                    end="",
                    flush=True,
                )
                if db_reporter is not None:
                    db_reporter.submit(
                        {
                            "FilesNumber": str(data_files_number),
                            "QueryName": queries_description[query_number + 1],
                            "IbisCommitHash": args.commit_ibis,
                            "FirstExecTimeMS": first_exec_time,
                            "WorstExecTimeMS": worst_exec_time,
                            "BestExecTimeMS": best_exec_time,
                            "AverageExecTimeMS": str(
                                queries_validation_results["q%s" % (query_number + 1)]
                            ),
                            "TotalTimeMS": total_exec_time,
                        }
                    )
    except IOError as err:
        print("Failed writing report file", args.r, err)
except Exception as exc:
    print("Failed: ", exc)
finally:
    if omnisci_server:
        omnisci_server.terminate()
