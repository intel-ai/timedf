if __name__ == "__main__":
    from braceexpand import braceexpand

    import mysql.connector
    import modin.pandas as pd
    import numpy as np
    import argparse
    import pathlib
    import time
    import glob
    import sys
    import os

    taxi_names = [
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

    # SELECT cab_type,
    #       count(*)
    # FROM trips
    # GROUP BY cab_type;
    # @hpat.jit fails with Invalid use of Function(<ufunc 'isnan'>) with argument(s) of type(s): (StringType), even when dtype is provided
    def q1(df):
        return df.groupby("cab_type").count()

    # SELECT passenger_count,
    #       count(total_amount)
    # FROM trips
    # GROUP BY passenger_count;
    def q2(df):
        return df.groupby("passenger_count", as_index=False).count()[
            ["passenger_count", "total_amount"]
        ]

    # SELECT passenger_count,
    #       EXTRACT(year from pickup_datetime) as year,
    #       count(*)
    # FROM trips
    # GROUP BY passenger_count,
    #         year;
    def q3(df):
        transformed = df.applymap(lambda x: x.year if hasattr(x, "year") else x)
        return transformed.groupby(
            ["pickup_datetime", "passenger_count"], as_index=False
        ).count()["passenger_count"]

    # SELECT passenger_count,
    #       EXTRACT(year from pickup_datetime) as year,
    #       round(trip_distance) distance,
    #       count(*) trips
    # FROM trips
    # GROUP BY passenger_count,
    #         year,
    #         distance
    # ORDER BY year,
    #         trips desc;
    def q4(df):
        transformed = df.applymap(
            lambda x: x.year
            if hasattr(x, "year")
            else round(x)
            if isinstance(x, (int, float)) and not np.isnan(x)
            else x
        )[["passenger_count", "pickup_datetime", "trip_distance"]].groupby(
            ["passenger_count", "pickup_datetime", "trip_distance"]
        )
        return (
            transformed.count()
            .reset_index()
            .sort_values(
                by=["pickup_datetime", "trip_distance"], ascending=[True, False]
            )
        )

    benchmarks = {"MQ01.pd": q1, "MQ02.pd": q2, "MQ03.pd": q3, "MQ04.pd": q4}

    # Load database reporting functions
    pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
    print(pathToReportDir)
    sys.path.insert(1, pathToReportDir)
    import report

    parser = argparse.ArgumentParser(description="Run NY Taxi benchmark using pandas")

    parser.add_argument("-r", default="report_pandas.csv", help="Report file name.")
    parser.add_argument(
        "-df",
        default=1,
        type=int,
        help="Number of datafiles to input into database for processing.",
    )
    parser.add_argument(
        "-dp", help="Wildcard pattern of datafiles that should be loaded."
    )
    parser.add_argument(
        "-i",
        dest="iterations",
        default=5,
        type=int,
        help="Number of iterations to run every benchmark. Best result is selected.",
    )

    parser.add_argument(
        "-db-server", default="localhost", help="Host name of MySQL server"
    )
    parser.add_argument(
        "-db-port", default=3306, type=int, help="Port number of MySQL server"
    )
    parser.add_argument(
        "-db-user",
        default="",
        help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db-pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database",
    )
    parser.add_argument(
        "-db-name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results",
    )
    parser.add_argument(
        "-db-table", help="Table to use to store results for this benchmark."
    )

    parser.add_argument(
        "-commit",
        default="1234567890123456789012345678901234567890",
        help="Commit hash to use to record this benchmark results",
    )

    args = parser.parse_args()

    if args.df <= 0:
        print("Bad number of data files specified", args.df)
        sys.exit(1)

    if args.iterations < 1:
        print("Bad number of iterations specified", args.t)

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
        db_reporter = report.DbReport(
            db,
            args.db_table,
            {
                "FilesNumber": "INT UNSIGNED NOT NULL",
                "FragmentSize": "BIGINT UNSIGNED NOT NULL",
                "BenchName": "VARCHAR(500) NOT NULL",
                "BestExecTimeMS": "BIGINT UNSIGNED",
                "BestTotalTimeMS": "BIGINT UNSIGNED",
                "WorstExecTimeMS": "BIGINT UNSIGNED",
                "WorstTotalTimeMS": "BIGINT UNSIGNED",
                "AverageExecTimeMS": "BIGINT UNSIGNED",
                "AverageTotalTimeMS": "BIGINT UNSIGNED",
            },
            {"ScriptName": "taxibench_pandas.py", "CommitHash": args.commit},
        )

    dataFileNames = list(braceexpand(args.dp))
    dataFileNames = sorted([x for f in dataFileNames for x in glob.glob(f)])
    if len(dataFileNames) == 0:
        print("Could not find any data files matching", args.dp)
        sys.exit(2)

    # print("READING", args.df, "DATAFILES")
    # dataFilesNumber = len(dataFileNames[: args.df])
    def read_datafile(f):
        print("READING DATAFILE", f)
        t1 = time.time()
        result = pd.read_csv(
            f,
            header=None,
            names=taxi_names,
            parse_dates=["pickup_datetime", "dropoff_datetime",],
        )
        t2 = time.time()
        print("Time to load {} datafile: {}".format(f, t2 - t1))
        return result

    # df_from_each_file = [read_datafile(f) for f in dataFileNames[:args.df]]
    df_from_each_file = [read_datafile(f) for f in dataFileNames]

    t1 = time.time()
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    t2 = time.time()
    print("Time to concatenated dataframes: {}".format(t2 - t1))

    try:
        with open(args.r, "w") as report:
            for benchName, query in benchmarks.items():
                bestExecTime = float("inf")
                for iii in range(1, args.iterations + 1):
                    print(
                        "RUNNING BENCHMARK NUMBER", benchName, "ITERATION NUMBER", iii
                    )
                    query_df = concatenated_df
                    t1 = time.time()
                    query(query_df)
                    t2 = time.time()
                    ttt = int(round((t2 - t1) * 1000))
                    if bestExecTime > ttt:
                        bestExecTime = ttt
                print("BENCHMARK", benchName, "EXEC TIME", bestExecTime)
                print(
                    0,
                    ",",
                    0,
                    ",",
                    benchName,
                    ",",
                    bestExecTime,
                    ",",
                    bestExecTime,
                    ",",
                    bestExecTime,
                    ",",
                    bestExecTime,
                    ",",
                    bestExecTime,
                    ",",
                    bestExecTime,
                    ",",
                    "",
                    "\n",
                    file=report,
                    sep="",
                    end="",
                    flush=True,
                )
                db_reporter = None
                if db_reporter is not None:
                    db_reporter.submit(
                        {
                            "FilesNumber": 0,
                            "FragmentSize": 0,
                            "BenchName": benchName,
                            "BestExecTimeMS": bestExecTime,
                            "BestTotalTimeMS": bestExecTime,
                            "WorstExecTimeMS": bestExecTime,
                            "WorstTotalTimeMS": bestExecTime,
                            "AverageExecTimeMS": bestExecTime,
                            "AverageTotalTimeMS": bestExecTime,
                        }
                    )
    except IOError as err:
        print("Failed writing report file", args.r, err)
