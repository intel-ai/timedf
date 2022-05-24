# Original SQL queries can be found here https://tech.marksblogg.com/billion-nyc-taxi-rides-nvidia-pascal-titan-x-mapd.html
from timeit import default_timer as timer

import pandas as pd

from utils import (
    files_names_from_pattern,
    import_pandas_into_module_namespace,
    load_data_pandas,
    load_data_modin_on_omnisci,
    print_results,
    get_ny_taxi_dataset_size,
    check_support,
)

accepted_data_files_for_pandas_import_mode = ["trips_xaa", "trips_xab", "trips_xac"]


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
# SELECT cab_type,
#       count(*)
# FROM trips
# GROUP BY cab_type;
# @hpat.jit fails with Invalid use of Function(<ufunc 'isnan'>) with argument(s) of type(s): (StringType), even when dtype is provided
def q1(df, pandas_mode):
    t0 = timer()
    if pandas_mode != "Modin_on_omnisci":
        q1_output = df.groupby("cab_type")["cab_type"].count()
    else:
        q1_output = df.groupby("cab_type").size()
        q1_output.shape  # to trigger real execution
    query_time = timer() - t0

    return query_time, q1_output


# SELECT passenger_count,
#       avg(total_amount)
# FROM trips
# GROUP BY passenger_count;
def q2(df, pandas_mode):
    t0 = timer()
    if pandas_mode != "Modin_on_omnisci":
        q2_output = df.groupby("passenger_count", as_index=False).mean()[
            ["passenger_count", "total_amount"]
        ]
    else:
        q2_output = df.groupby("passenger_count").agg({"total_amount": "mean"})
        q2_output.shape  # to trigger real execution
    query_time = timer() - t0

    return query_time, q2_output


# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       count(*)
# FROM trips
# GROUP BY passenger_count,
#         pickup_year;
def q3(df, pandas_mode):
    t0 = timer()
    if pandas_mode != "Modin_on_omnisci":
        transformed = pd.DataFrame(
            {
                "passenger_count": df["passenger_count"],
                "pickup_datetime": df["pickup_datetime"].dt.year,
            }
        )
        q3_output = (
            transformed.groupby(["pickup_datetime", "passenger_count"]).size().reset_index()
        )
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.year
        q3_output = df.groupby(["passenger_count", "pickup_datetime"]).size()
        q3_output.shape  # to trigger real execution
    query_time = timer() - t0

    return query_time, q3_output


# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       cast(trip_distance as int) AS distance,
#       count(*) AS the_count
# FROM trips
# GROUP BY passenger_count,
#         pickup_year,
#         distance
# ORDER BY pickup_year,
#         the_count desc;

# SQL query with sorting for results validation
# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       cast(trip_distance as int) AS distance,
#       count(*) AS the_count
# FROM agent_test_modin
# GROUP BY passenger_count,
#         pickup_year,
#         distance
# ORDER BY passenger_count, pickup_year, distance, the_count;
def q4(df, pandas_mode):
    t0 = timer()
    if pandas_mode != "Modin_on_omnisci":
        transformed = pd.DataFrame(
            {
                "passenger_count": df["passenger_count"],
                "pickup_datetime": df["pickup_datetime"].dt.year,
                "trip_distance": df["trip_distance"].astype("int64"),
            }
        )
        q4_output = (
            transformed.groupby(["passenger_count", "pickup_datetime", "trip_distance"])
            .size()
            .reset_index()
            .sort_values(by=["pickup_datetime", 0], ascending=[True, False])
        )
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.year
        df["trip_distance"] = df["trip_distance"].astype("int64")
        q4_output = (
            df.groupby(["passenger_count", "pickup_datetime", "trip_distance"], sort=False)
            .size()
            .reset_index()
            .sort_values(by=["pickup_datetime", 0], ignore_index=True, ascending=[True, False])
        )
        q4_output.shape  # to trigger real execution
    query_time = timer() - t0

    return query_time, q4_output


def etl(filename, files_limit, columns_names, columns_types, output_for_validation, pandas_mode):

    if pandas_mode == "Modin_on_omnisci" and any(f.endswith(".gz") for f in filename):
        raise NotImplementedError(
            "Modin_on_omnisci mode doesn't support import of compressed files yet"
        )

    queries = {"Query1": q1, "Query2": q2, "Query3": q3, "Query4": q4}
    etl_results = {x: 0.0 for x in queries.keys()}

    t0 = timer()
    if pandas_mode == "Modin_on_omnisci":
        df_from_each_file = [
            load_data_modin_on_omnisci(
                filename=f,
                columns_names=columns_names,
                columns_types=columns_types,
                parse_dates=["timestamp"],
                pd=run_benchmark.__globals__["pd"],
            )
            for f in filename
        ]
    else:
        df_from_each_file = [
            load_data_pandas(
                filename=f,
                columns_names=columns_names,
                header=None,
                nrows=None,
                use_gzip=f.endswith(".gz"),
                parse_dates=["pickup_datetime", "dropoff_datetime"],
                pd=run_benchmark.__globals__["pd"],
                pandas_mode=pandas_mode,
            )
            for f in filename
        ]

    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    # this is to trigger data import in `MOdin_on_omnisci` mode
    if pandas_mode == "Modin_on_omnisci":
        try:
            from modin.experimental.core.execution.native.implementations.omnisci_on_native.db_worker import (
                DbWorker,
            )

            concatenated_df.shape
            concatenated_df._query_compiler._modin_frame._partitions[0][
                0
            ].frame_id = DbWorker().import_arrow_table(
                concatenated_df._query_compiler._modin_frame._partitions[0][0].get()
            )
        except ModuleNotFoundError: # fallback to omniscidbe4py package
            from modin.experimental.core.execution.native.implementations.omnisci_on_native.omnisci_worker import (
                OmnisciServer,
            )

            concatenated_df.shape
            concatenated_df._query_compiler._modin_frame._partitions[0][
                0
            ].frame_id = OmnisciServer().put_arrow_to_omnisci(
                concatenated_df._query_compiler._modin_frame._partitions[0][0].get()
            )

    etl_results["t_readcsv"] = timer() - t0

    queries_parameters = {
        query_name: {
            "df": concatenated_df.copy() if pandas_mode == "Modin_on_omnisci" else concatenated_df,
            "pandas_mode": pandas_mode,
        }
        for query_name in list(queries.keys())
    }

    return run_queries(
        queries=queries,
        parameters=queries_parameters,
        etl_results=etl_results,
        output_for_validation=output_for_validation,
    )


def run_benchmark(parameters):
    check_support(parameters, unsupported_params=["optimizer", "no_ml", "gpu_memory"])

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
        "category",
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
        "category",
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

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )

    etl_results = None
    pd_queries_outputs = {} if parameters["validation"] else None

    pandas_files_limit = parameters["dfiles_num"]
    filename = files_names_from_pattern(parameters["data_file"])[:pandas_files_limit]
    etl_results = etl(
        filename=filename,
        files_limit=pandas_files_limit,
        columns_names=columns_names,
        columns_types=columns_types,
        output_for_validation=pd_queries_outputs,
        pandas_mode=parameters["pandas_mode"],
    )

    print_results(results=etl_results, backend=parameters["pandas_mode"], unit="s")
    etl_results["Backend"] = parameters["pandas_mode"]
    etl_results["dfiles_num"] = parameters["dfiles_num"]
    etl_results["dataset_size"] = get_ny_taxi_dataset_size(parameters["dfiles_num"])

    return {"ETL": [etl_results], "ML": []}
