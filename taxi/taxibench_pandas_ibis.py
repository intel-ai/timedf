import os
import sys
import traceback
import warnings
from timeit import default_timer as timer

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (  # noqa: F401 ("compare_dataframes" imported, but unused. Used in commented code.)
    check_fragments_size,
    compare_dataframes,
    files_names_from_pattern,
    import_pandas_into_module_namespace,
    load_data_pandas,
    print_results,
    write_to_csv_by_chunks,
)  # noqa: E402 (module level import not at top of file. Import after sys.path)


def validation_prereqs(omnisci_server_worker, data_files_names, files_limit, columns_names):
    return omnisci_server_worker.import_data_by_pandas(
        data_files_names=data_files_names, files_limit=files_limit, columns_names=columns_names,
    )


def run_queries(queries, parameters, etl_times):
    for query_number, (query_name, query_func) in enumerate(queries.items()):
        exec_time = query_func(**parameters)
        etl_times[query_name] = exec_time
    return etl_times


# Queries definitions
def q1_ibis(table, df_pandas, queries_validation_results, queries_validation_flags, validation):
    t_query = 0
    t0 = timer()
    q1_output_ibis = (  # noqa: F841 (assigned, but unused. Used in commented code.)
        table.groupby("cab_type").count().sort_by("cab_type")["cab_type", "count"].execute()
    )
    t_query += timer() - t0

    if validation and not queries_validation_flags["q1"]:
        print("Validating query 1 results ...")

        queries_validation_flags["q1"] = True

        q1_output_pd = df_pandas.groupby("cab_type")["cab_type"].count()

        # Casting of Pandas q1 output to Pandas.DataFrame type, which is compartible with
        # Ibis q1 output
        q1_output_pd_df = q1_output_pd.to_frame()
        q1_output_pd_df.loc[:, "count"] = q1_output_pd_df.loc[:, "cab_type"]
        q1_output_pd_df["cab_type"] = q1_output_pd_df.index
        q1_output_pd_df.index = [i for i in range(len(q1_output_pd_df))]

        # queries_validation_results["q1"] = compare_dataframes(
        #     ibis_df=q1_output_pd_df,
        #     pandas_df=q1_output_ibis,
        #     pd=run_benchmark.__globals__["pd"],
        # )
        if queries_validation_results["q1"]:
            print("q1 results are validated!")

    return t_query


def q2_ibis(table, df_pandas, queries_validation_results, queries_validation_flags, validation):
    t_query = 0
    t0 = timer()
    q2_output_ibis = (  # noqa: F841 (assigned, but unused. Used in commented code.)
        table.groupby("passenger_count")
        .aggregate(total_amount=table.total_amount.mean())[["passenger_count", "total_amount"]]
        .execute()
    )
    t_query += timer() - t0

    if validation and not queries_validation_flags["q2"]:
        print("Validating query 2 results ...")

        queries_validation_flags["q2"] = True

        q2_output_pd = df_pandas.groupby(  # noqa: F841 (assigned, but unused. Used in commented code.)
            "passenger_count", as_index=False
        ).mean()[
            ["passenger_count", "total_amount"]
        ]

        # queries_validation_results["q2"] = compare_dataframes(
        #     pandas_df=q2_output_pd,
        #     ibis_df=q2_output_ibis,
        #     pd=run_benchmark.__globals__["pd"],
        # )
        if queries_validation_results["q2"]:
            print("q2 results are validated!")

    return t_query


def q3_ibis(table, df_pandas, queries_validation_results, queries_validation_flags, validation):
    t_query = 0
    t0 = timer()
    q3_output_ibis = (  # noqa: F841 (assigned, but unused. Used in commented code.)
        table.groupby(
            [table.passenger_count, table.pickup_datetime.year().name("pickup_datetime")]
        )
        .aggregate(count=table.passenger_count.count())
        .execute()
    )
    t_query += timer() - t0

    if validation and not queries_validation_flags["q3"]:
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
        q3_output_pd_df["passenger_count"] = q3_output_pd.index.droplevel(level="pickup_datetime")
        q3_output_pd_df["pickup_datetime"] = q3_output_pd.index.droplevel(level="passenger_count")
        q3_output_pd_df = q3_output_pd_df.astype({"pickup_datetime": "int32"})
        q3_output_pd_df.loc[:, "count"] = count_df
        q3_output_pd_df.index = [i for i in range(len(q3_output_pd_df))]

        # queries_validation_results["q3"] = compare_dataframes(
        #     pandas_df=q3_output_pd_df,
        #     ibis_df=q3_output_ibis,
        #     pd=run_benchmark.__globals__["pd"],
        # )
        if queries_validation_results["q3"]:
            print("q3 results are validated!")

    return t_query


def q4_ibis(table, df_pandas, queries_validation_results, queries_validation_flags, validation):
    t_query = 0
    t0 = timer()
    q4_ibis_sized = table.groupby(
        [
            table.passenger_count,
            table.pickup_datetime.year().name("pickup_datetime"),
            table.trip_distance.cast("int64").name("trip_distance"),
        ]
    ).size()
    q4_output_ibis = q4_ibis_sized.sort_by(  # noqa: F841 (assigned, but unused. Used in commented code.)
        [("pickup_datetime", True), ("count", False)]
    ).execute()
    t_query += timer() - t0

    if validation and not queries_validation_flags["q4"]:
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

        q4_output_pd = q4_pd_sized.sort_values(by=["pickup_datetime", 0], ascending=[True, False])

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
        # compare_result_1 = compare_dataframes(
        #     pandas_df=q4_output_pd["pickup_datetime"],
        #     ibis_df=q4_output_ibis["pickup_datetime"],
        #     pd=run_benchmark.__globals__["pd"],
        # )
        # compare_result_2 = compare_dataframes(
        #     pandas_df=q4_output_pd["count"],
        #     ibis_df=q4_output_ibis["count"],
        #     pd=run_benchmark.__globals__["pd"],
        # )

        # compare_result_3 is the result of q4 output table all elements presence check
        q4_output_ibis_validation = q4_ibis_sized.sort_by(  # noqa: F841 (assigned, but unused. Used in commented code.)
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

        # compare_result_3 = compare_dataframes(
        #     pandas_df=q4_output_pd_valid,
        #     ibis_df=q4_output_ibis_validation,
        #     pd=run_benchmark.__globals__["pd"],
        # )

        queries_validation_results["q4"] = (
            compare_result_1  # noqa: F821 (undefined name. Defined in commented code.)
            and compare_result_2  # noqa: F821 (undefined name. Defined in commented code.)
            and compare_result_3  # noqa: F821 (undefined name. Defined in commented code.)
        )
        if queries_validation_results["q4"]:
            print("q4 results are validated!")

    return t_query


def etl_ibis(
    filename,
    files_limit,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    validation,
    import_mode,
    fragments_size,
):
    import ibis

    fragments_size = check_fragments_size(fragments_size, count_table=1, import_mode=import_mode)

    queries = {
        "Query1": q1_ibis,
        "Query2": q2_ibis,
        "Query3": q3_ibis,
        "Query4": q4_ibis,
    }
    etl_times = {x: 0.0 for x in queries.keys()}
    etl_times["t_readcsv"] = 0.0
    etl_times["t_connect"] = 0.0

    queries_validation_results = {"q%s" % i: False for i in range(1, 5)}
    queries_validation_flags = {"q%s" % i: False for i in range(1, 5)}

    omnisci_server_worker.connect_to_server()

    data_files_names = files_names_from_pattern(filename)

    if len(data_files_names) == 0:
        raise FileNotFoundError(f"Could not find any data files matching: [{filename}]")

    data_files_extension = data_files_names[0].split(".")[-1]
    if not all([name.endswith(data_files_extension) for name in data_files_names]):
        raise NotImplementedError(
            "Import of data files with different extensions is not supported"
        )

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)

    # Create table and import data for ETL queries
    if create_new_table:
        schema_table = ibis.Schema(names=columns_names, types=columns_types)
        if import_mode == "copy-from":
            t0 = timer()
            omnisci_server_worker.create_table(
                table_name=table_name,
                schema=schema_table,
                database=database_name,
                fragment_size=fragments_size[0],
            )
            etl_times["t_connect"] += timer() - t0
            table_import = omnisci_server_worker.database(database_name).table(table_name)
            etl_times["t_connect"] += omnisci_server_worker.get_conn_creation_time()

            for file_to_import in data_files_names[:files_limit]:
                t0 = timer()
                table_import.read_csv(file_to_import, header=False, quotechar='"', delimiter=",")
                etl_times["t_readcsv"] += timer() - t0

        elif import_mode == "pandas":
            t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
                table_name=table_name,
                data_files_names=data_files_names,
                files_limit=files_limit,
                columns_names=columns_names,
                columns_types=columns_types,
                header=None,
                nrows=None,
                compression_type="gzip" if data_files_extension == "gz" else None,
                use_columns_types_for_pd=False,
            )

            etl_times["t_readcsv"] = t_import_pandas + t_import_ibis
            etl_times["t_connect"] = omnisci_server_worker.get_conn_creation_time()

        elif import_mode == "fsi":
            data_file_path = None
            if data_files_extension == "gz" or len(data_files_names) > 1:
                data_file_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "tmp")
                )
                data_file_path = os.path.join(
                    data_file_dir, f"taxibench-{files_limit}-files-fsi.csv"
                )

            if data_file_path and not os.path.exists(data_file_path):
                if not os.path.exists(data_file_dir):
                    os.mkdir(data_file_dir)
                try:
                    for file_name in data_files_names[:files_limit]:
                        write_to_csv_by_chunks(
                            file_to_write=file_name, output_file=data_file_path, write_mode="ab",
                        )
                except Exception:
                    os.remove(data_file_path)
                    raise

            t0 = timer()
            omnisci_server_worker.get_conn().create_table_from_csv(
                table_name,
                data_file_path if data_file_path else data_files_names[0],
                schema_table,
                header=False,
                fragment_size=fragments_size[0],
            )
            etl_times["t_readcsv"] += timer() - t0
            etl_times["t_connect"] = omnisci_server_worker.get_conn_creation_time()

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    etl_times["t_connect"] += omnisci_server_worker.get_conn_creation_time()
    t0 = timer()
    table = omnisci_server_worker.database(database_name).table(table_name)
    etl_times["t_connect"] += timer() - t0

    df_pandas = None
    if validation:
        df_pandas = validation_prereqs(
            omnisci_server_worker, data_files_names, files_limit, columns_names
        )

    queries_parameters = {
        "table": table,
        "df_pandas": df_pandas,
        "queries_validation_results": queries_validation_results,
        "queries_validation_flags": queries_validation_flags,
        "validation": validation,
    }
    return run_queries(queries=queries, parameters=queries_parameters, etl_times=etl_times)


# SELECT cab_type,
#       count(*)
# FROM trips
# GROUP BY cab_type;
# @hpat.jit fails with Invalid use of Function(<ufunc 'isnan'>) with argument(s) of type(s): (StringType), even when dtype is provided
def q1_pandas(df):
    t0 = timer()
    df.groupby("cab_type").count()
    return timer() - t0


# SELECT passenger_count,
#       count(total_amount)
# FROM trips
# GROUP BY passenger_count;
def q2_pandas(df):
    t0 = timer()
    df.groupby("passenger_count", as_index=False).count()[["passenger_count", "total_amount"]]
    return timer() - t0


# SELECT passenger_count,
#       EXTRACT(year from pickup_datetime) as year,
#       count(*)
# FROM trips
# GROUP BY passenger_count,
#         year;
def q3_pandas(df):
    t0 = timer()
    transformed = df.applymap(lambda x: x.year if hasattr(x, "year") else x)
    transformed.groupby(["pickup_datetime", "passenger_count"], as_index=False).count()[
        "passenger_count"
    ]
    return timer() - t0


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
def q4_pandas(df):
    t0 = timer()
    transformed = df.applymap(
        lambda x: x.year
        if hasattr(x, "year")
        else round(x)
        if isinstance(x, (int, float)) and not np.isnan(x)
        else x
    )[["passenger_count", "pickup_datetime", "trip_distance"]].groupby(
        ["passenger_count", "pickup_datetime", "trip_distance"]
    )
    (
        transformed.count()
        .reset_index()
        .sort_values(by=["pickup_datetime", "trip_distance"], ascending=[True, False])
    )
    return timer() - t0


def etl_pandas(
    filename, files_limit, columns_names, columns_types,
):
    queries = {
        "Query1": q1_pandas,
        "Query2": q2_pandas,
        "Query3": q3_pandas,
        "Query4": q4_pandas,
    }
    etl_times = {x: 0.0 for x in queries.keys()}

    t0 = timer()
    df_from_each_file = [
        load_data_pandas(
            filename=f,
            columns_names=columns_names,
            header=None,
            nrows=None,
            use_gzip=f.endswith(".gz"),
            parse_dates=["pickup_datetime", "dropoff_datetime"],
            pd=run_benchmark.__globals__["pd"],
        )
        for f in filename
    ]
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    etl_times["t_readcsv"] = timer() - t0

    queries_parameters = {"df": concatenated_df}
    return run_queries(queries=queries, parameters=queries_parameters, etl_times=etl_times)


def run_benchmark(parameters):

    ignored_parameters = {
        "optimizer": parameters["optimizer"],
        "no_ml": parameters["no_ml"],
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are ignored", RuntimeWarning)

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    columns_names = [
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

    columns_types = [
        "int64",
        "int64",
        "timestamp",
        "timestamp",
        "category",
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
        "category",
        "category",
        "category",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "category",
        "float64",
        "float64",
        "category",
        "category",
        "category",
        "float64",
        "float64",
        "float64",
        "float64",
        "category",
        "float64",
        "float64",
        "category",
        "category",
        "category",
        "float64",
    ]

    if parameters["dfiles_num"] <= 0:
        print("Bad number of data files specified: ", parameters["dfiles_num"])
        sys.exit(1)
    try:
        if not parameters["no_pandas"]:
            import_pandas_into_module_namespace(
                namespace=run_benchmark.__globals__,
                mode=parameters["pandas_mode"],
                ray_tmpdir=parameters["ray_tmpdir"],
                ray_memory=parameters["ray_memory"],
            )

        etl_times_ibis = None
        etl_times = None
        if not parameters["no_ibis"]:
            etl_times_ibis = etl_ibis(
                filename=parameters["data_file"],
                files_limit=parameters["dfiles_num"],
                columns_names=columns_names,
                columns_types=columns_types,
                database_name=parameters["database_name"],
                table_name=parameters["table"],
                omnisci_server_worker=parameters["omnisci_server_worker"],
                delete_old_database=not parameters["dnd"],
                ipc_connection=parameters["ipc_connection"],
                create_new_table=not parameters["dni"],
                validation=parameters["validation"],
                import_mode=parameters["import_mode"],
                fragments_size=parameters["fragments_size"],
            )

            print_results(results=etl_times_ibis, backend="Ibis", unit="ms")
            etl_times_ibis["Backend"] = "Ibis"
            etl_times_ibis["dfiles_num"] = parameters["dfiles_num"]

        if not parameters["no_pandas"]:
            pandas_files_limit = parameters["dfiles_num"]
            filename = files_names_from_pattern(parameters["data_file"])[:pandas_files_limit]
            etl_times = etl_pandas(
                filename=filename,
                files_limit=pandas_files_limit,
                columns_names=columns_names,
                columns_types=columns_types,
            )

            print_results(results=etl_times, backend=parameters["pandas_mode"], unit="ms")
            etl_times["Backend"] = parameters["pandas_mode"]
            etl_times["dfiles_num"] = parameters["dfiles_num"]

        return {"ETL": [etl_times_ibis, etl_times], "ML": []}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
