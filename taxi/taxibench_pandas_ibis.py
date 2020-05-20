import os
import sys
import traceback
import warnings
from timeit import default_timer as timer

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    check_fragments_size,
    compare_dataframes,
    files_names_from_pattern,
    import_pandas_into_module_namespace,
    load_data_pandas,
    print_results,
    write_to_csv_by_chunks,
    get_dir,
    get_ny_taxi_dataset_size,
)


def run_queries(queries, parameters, etl_results, output_for_validation=None):
    for query_name, query_func in queries.items():
        query_result = query_func(**parameters[query_name])
        etl_results[query_name] = (
            query_result[0]
            if isinstance(query_result, (tuple, list)) and len(query_result) == 2
            else query_result
        )
        if output_for_validation is not None:
            assert len(query_result) == 2
            output_for_validation[query_name] = query_result[1]

    return etl_results


# Queries definitions
def q1_ibis(table, input_for_validation):
    t_query = 0
    t0 = timer()
    q1_output_ibis = (
        table.groupby("cab_type").count().sort_by("cab_type")["cab_type", "count"].execute()
    )
    t_query += timer() - t0

    if input_for_validation:
        print("Validating query 1 results ...")

        q1_output_pd = input_for_validation["Query1"]

        # Casting of Pandas q1 output to Pandas.DataFrame type, which is compartible with
        # Ibis q1 output
        q1_output_pd_data = {
            q1_output_pd.name: q1_output_pd.index.to_numpy(),
            "count": q1_output_pd.to_numpy(),
        }
        q1_output_pd_df = pd.DataFrame(q1_output_pd_data, columns=[q1_output_pd.name, "count"])
        q1_output_pd_df = q1_output_pd_df.astype({"cab_type": "category"}, copy=False)

        compare_dataframes(
            ibis_dfs=[q1_output_pd_df], pandas_dfs=[q1_output_ibis], sort_cols=[], drop_cols=[]
        )

    return t_query


def q2_ibis(table, input_for_validation):
    t_query = 0
    t0 = timer()
    q2_output_ibis = (
        table.groupby("passenger_count")
        .aggregate(total_amount=table.total_amount.count())[["passenger_count", "total_amount"]]
        .execute()
    )
    t_query += timer() - t0

    if input_for_validation is not None:
        print("Validating query 2 results ...")

        q2_output_pd = input_for_validation["Query2"]

        compare_dataframes(
            ibis_dfs=[q2_output_pd], pandas_dfs=[q2_output_ibis], sort_cols=[], drop_cols=[]
        )

    return t_query


def q3_ibis(table, input_for_validation):
    t_query = 0
    t0 = timer()
    q3_output_ibis = (
        table.groupby(
            [table.passenger_count, table.pickup_datetime.year().name("pickup_datetime"),]
        )
        .aggregate(count=table.passenger_count.count())
        .execute()
    )
    t_query += timer() - t0

    if input_for_validation is not None:
        print("Validating query 3 results ...")

        q3_output_pd = input_for_validation["Query3"]
        # Casting of Pandas q3 output to Pandas.DataFrame type, which is compartible with
        # Ibis q3 output
        passenger_count_col = q3_output_pd.index.droplevel(level="pickup_datetime")
        pickup_datetime_col = q3_output_pd.index.droplevel(level="passenger_count")
        count_col = q3_output_pd[("passenger_count", "count")].to_numpy()
        q3_output_pd_casted = pd.DataFrame(
            {
                "passenger_count": passenger_count_col,
                "pickup_datetime": pickup_datetime_col,
                "count": count_col,
            }
        )

        compare_dataframes(
            ibis_dfs=[q3_output_pd_casted], pandas_dfs=[q3_output_ibis], sort_cols=[], drop_cols=[]
        )

    return t_query


def q4_ibis(table, input_for_validation):
    t_query = 0
    t0 = timer()
    q4_ibis_sized = table.groupby(
        [
            table.passenger_count,
            table.pickup_datetime.year().name("pickup_datetime"),
            table.trip_distance.round().cast("int64").name("trip_distance"),
        ]
    ).size()
    q4_output_ibis = q4_ibis_sized.sort_by([("pickup_datetime", True), ("count", False)]).execute()
    t_query += timer() - t0

    if input_for_validation is not None:
        print("Validating query 4 results ...")

        q4_output_pd = input_for_validation["Query4"]

        # Casting of Pandas q4 output to Pandas.DataFrame type, which is compartible with
        # Ibis q4 output
        q4_output_pd = q4_output_pd.rename(columns={0: "count"})
        q4_output_ibis = q4_output_ibis.sort_values(
            by=["passenger_count", "pickup_datetime", "trip_distance"],
            ascending=[True, True, True],
        )
        q4_output_pd = q4_output_pd.sort_values(
            by=["passenger_count", "pickup_datetime", "trip_distance"],
            ascending=[True, True, True],
        )

        compare_dataframes(
            ibis_dfs=[q4_output_ibis], pandas_dfs=[q4_output_pd], sort_cols=[], drop_cols=[]
        )

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
    input_for_validation,
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
    etl_results = {x: 0.0 for x in queries.keys()}
    etl_results["t_readcsv"] = 0.0
    etl_results["t_connect"] = 0.0

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
            etl_results["t_connect"] += timer() - t0
            table_import = omnisci_server_worker.database(database_name).table(table_name)
            etl_results["t_connect"] += omnisci_server_worker.get_conn_creation_time()

            for file_to_import in data_files_names[:files_limit]:
                t0 = timer()
                table_import.read_csv(file_to_import, header=False, quotechar='"', delimiter=",")
                etl_results["t_readcsv"] += timer() - t0

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

            etl_results["t_readcsv"] = t_import_pandas + t_import_ibis
            etl_results["t_connect"] = omnisci_server_worker.get_conn_creation_time()

        elif import_mode == "fsi":
            data_file_path = None
            # If data files are compressed or number of csv files is more than one,
            # data files (or single compressed file) should be transformed to single csv file.
            # Before files transformation, script checks existance of already transformed file
            # in the directory passed with -data_file flag, and then, if file is not found,
            # in the omniscripts/tmp directory.
            if data_files_extension == "gz" or len(data_files_names) > 1:
                data_file_path = os.path.abspath(
                    os.path.join(
                        os.path.dirname(data_files_names[0]),
                        f"taxibench-{files_limit}-files-fsi.csv",
                    )
                )
                data_file_tmp_dir = os.path.join(get_dir("repository_root"), "tmp")
                if not os.path.exists(data_file_path):
                    data_file_path = os.path.join(
                        data_file_tmp_dir, f"taxibench-{files_limit}-files-fsi.csv"
                    )
                    data_file_dir = data_file_tmp_dir

            if data_file_path and not os.path.exists(data_file_path):
                if not os.path.exists(data_file_tmp_dir):
                    os.mkdir(data_file_tmp_dir)
                try:
                    for file_name in data_files_names[:files_limit]:
                        write_to_csv_by_chunks(
                            file_to_write=file_name, output_file=data_file_path, write_mode="ab",
                        )
                except Exception as exc:
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
            etl_results["t_readcsv"] += timer() - t0
            etl_results["t_connect"] = omnisci_server_worker.get_conn_creation_time()

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    etl_results["t_connect"] += omnisci_server_worker.get_conn_creation_time()
    t0 = timer()
    table = omnisci_server_worker.database(database_name).table(table_name)
    etl_results["t_connect"] += timer() - t0

    queries_parameters = {
        query_name: {"table": table, "input_for_validation": input_for_validation}
        for query_name in queries.keys()
    }
    return run_queries(queries=queries, parameters=queries_parameters, etl_results=etl_results)


# SELECT cab_type,
#       count(*)
# FROM trips
# GROUP BY cab_type;
# @hpat.jit fails with Invalid use of Function(<ufunc 'isnan'>) with argument(s) of type(s): (StringType), even when dtype is provided
def q1_pandas(df):
    t0 = timer()
    q1_pandas_output = df.groupby("cab_type")["cab_type"].count()
    query_time = timer() - t0

    return query_time, q1_pandas_output


# SELECT passenger_count,
#       count(total_amount)
# FROM trips
# GROUP BY passenger_count;
def q2_pandas(df):
    t0 = timer()
    q2_pandas_output = df.groupby("passenger_count", as_index=False).count()[
        ["passenger_count", "total_amount"]
    ]
    query_time = timer() - t0

    return query_time, q2_pandas_output


# SELECT passenger_count,
#       EXTRACT(year from pickup_datetime) as year0,
#       count(*)
# FROM trips
# GROUP BY passenger_count,
#         year0;
def q3_pandas(df):
    t0 = timer()
    transformed = pd.DataFrame(
        {
            "passenger_count": df["passenger_count"],
            "pickup_datetime": df["pickup_datetime"].dt.year,
        }
    )
    q3_pandas_output = transformed.groupby(["pickup_datetime", "passenger_count"]).agg(
        {"passenger_count": ["count"]}
    )
    query_time = timer() - t0

    return query_time, q3_pandas_output


# SELECT passenger_count,
#       EXTRACT(year from pickup_datetime) as year0,
#       round(trip_distance) distance,
#       count(*) trips
# FROM trips
# GROUP BY passenger_count,
#         year0,
#         distance
# ORDER BY year0,
#         trips desc;
def q4_pandas(df, validation):
    t0 = timer()
    transformed = pd.DataFrame(
        {
            "passenger_count": df["passenger_count"],
            "pickup_datetime": df["pickup_datetime"].dt.year,
            "trip_distance": df["trip_distance"].round(),
        }
    )
    q4_pandas_output = (
        transformed.groupby(["passenger_count", "pickup_datetime", "trip_distance"]).size()
        .reset_index()
        .sort_values(by=["pickup_datetime", 0], ascending=[True, False])
    ).astype({"trip_distance": "int64"}, copy=False)
    query_time = timer() - t0

    if validation:
        import math
        transformed["trip_distance"] = df["trip_distance"].transform(lambda x: round(x)
                if round(math.modf(x)[0], 5) != 0.5
                else math.modf(x + 1)[1])
        q4_pandas_output = (
            transformed.groupby(["passenger_count", "pickup_datetime", "trip_distance"]).size()
            .reset_index()
            .sort_values(by=["pickup_datetime", 0], ascending=[True, False])
        ).astype({"trip_distance": "int64"}, copy=False)

    return query_time, q4_pandas_output


def etl_pandas(
    filename, files_limit, columns_names, columns_types, output_for_validation,
):
    queries = {
        "Query1": q1_pandas,
        "Query2": q2_pandas,
        "Query3": q3_pandas,
        "Query4": q4_pandas,
    }
    etl_results = {x: 0.0 for x in queries.keys()}

    t0 = timer()
    df_from_each_file = [
        load_data_pandas(
            filename=f,
            columns_names=columns_names,
            header=None,
            nrows=None,
            use_gzip=f.endswith(".gz"),
            parse_dates=["pickup_datetime", "dropoff_datetime",],
            pd=run_benchmark.__globals__["pd"],
        )
        for f in filename
    ]
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    etl_results["t_readcsv"] = timer() - t0

    queries_parameters = {
        query_name: {"df": concatenated_df} for query_name in list(queries.keys())[:3]
    }
    queries_parameters.update(
        {
            "Query4": {
                "df": concatenated_df,
                "validation": True if output_for_validation is not None else False,
            }
        }
    )
    return run_queries(
        queries=queries,
        parameters=queries_parameters,
        etl_results=etl_results,
        output_for_validation=output_for_validation,
    )


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
        raise ValueError(f"Bad number of data files specified: {parameters['dfiles_num']}")
    try:
        if not parameters["no_pandas"]:
            import_pandas_into_module_namespace(
                namespace=run_benchmark.__globals__,
                mode=parameters["pandas_mode"],
                ray_tmpdir=parameters["ray_tmpdir"],
                ray_memory=parameters["ray_memory"],
            )

        etl_results_ibis = None
        etl_results = None
        pd_queries_outputs = {} if parameters["validation"] else None
        if not parameters["no_pandas"]:
            pandas_files_limit = parameters["dfiles_num"]
            filename = files_names_from_pattern(parameters["data_file"])[:pandas_files_limit]
            etl_results = etl_pandas(
                filename=filename,
                files_limit=pandas_files_limit,
                columns_names=columns_names,
                columns_types=columns_types,
                output_for_validation=pd_queries_outputs,
            )

            print_results(results=etl_results, backend=parameters["pandas_mode"], unit="ms")
            etl_results["Backend"] = parameters["pandas_mode"]

        if not parameters["no_ibis"]:
            etl_results_ibis = etl_ibis(
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
                input_for_validation=pd_queries_outputs,
                import_mode=parameters["import_mode"],
                fragments_size=parameters["fragments_size"],
            )

            print_results(results=etl_results_ibis, backend="Ibis", unit="ms")
            etl_results_ibis["Backend"] = "Ibis"
            etl_results_ibis["dfiles_num"] = parameters["dfiles_num"]
            etl_results_ibis["dataset_size"] = get_ny_taxi_dataset_size(parameters["dfiles_num"])

        return {"ETL": [etl_results_ibis, etl_results], "ML": []}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
