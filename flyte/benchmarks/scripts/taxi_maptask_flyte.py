import time
import typing

import pandas as pd
from flytekit import task, workflow, map_task, TaskMetadata, Resources


taxi_path = [
             'https://modin-datasets.s3.us-west-2.amazonaws.com/taxi/trips_xba.csv.gz',
             'https://modin-datasets.s3.us-west-2.amazonaws.com/taxi/trips_xbb.csv.gz',
             'https://modin-datasets.s3.us-west-2.amazonaws.com/taxi/trips_xbc.csv.gz',
             'https://modin-datasets.s3.us-west-2.amazonaws.com/taxi/trips_xbd.csv.gz'
            ]

taxi_path = ["https://modin-datasets.s3.amazonaws.com/taxi/trips_xaa_5M.csv.gz"]


cols = [
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
        "int",
        "str",
        "str",
        "str",
        "str",
        "int",
        "float",
        "float",
        "float",
        "float",
        "int",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
        "str",
        "float",
        "str",
        "str",
        "str",
        "float",
        "int",
        "float",
        "int",
        "int",
        "float",
        "float",
        "float",
        "float",
        "str",
        "float",
        "float",
        "str",
        "str",
        "str",
        "float",
        "float",
        "float",
        "float",
        "str",
        "float",
        "float",
        "str",
        "str",
        "str",
        "float",
    ]

dtypes = {c: t for c, t in zip(cols, columns_types)}

parse_dates = ["pickup_datetime", "dropoff_datetime"]

compression = 'gzip'


@task(requests=Resources(cpu="1", mem="32Gi"), limits=Resources(mem="128Gi"))
def get_taxi_dataset_task(
    data: str,
) -> pd.DataFrame:
    try:
        return pd.read_csv(data, compression=compression, names=cols, parse_dates=parse_dates)
    except:
        print(f'{data} contains incorrect columns')
        return pd.DataFrame()


@task
def taxi_q1_task(df: pd.DataFrame) -> str:
    print(df.columns)
    return pd.DataFrame(df.groupby(["cab_type"]).count()["trip_id"]).to_string()


@task
def taxi_q2_task(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("passenger_count", as_index=False).mean()[
        ["passenger_count", "total_amount"]
    ]


@task
def taxi_q3_task(df: pd.DataFrame) -> pd.DataFrame:
    res = df.groupby(["passenger_count", "pickup_datetime"]).size().reset_index()
    res.columns = res.columns.astype(str)
    return res


@task
def taxi_q4_task(df: pd.DataFrame) -> pd.DataFrame:
    transformed = pd.DataFrame(
        {
            "passenger_count": df["passenger_count"],
            "pickup_datetime": df["pickup_datetime"].dt.year,
            "trip_distance": df["trip_distance"].astype("int64"),
        }
    )
    transformed = (
        transformed.groupby(["passenger_count", "pickup_datetime", "trip_distance"])
        .size()
        .reset_index()
        .sort_values(by=["pickup_datetime", 0], ascending=[True, False])
    )
    transformed.columns = transformed.columns.astype(str)
    return transformed


@workflow
def taxi_wf(
    taxi_path: typing.List[str] = taxi_path,
) -> str:
    df = map_task(get_taxi_dataset_task, metadata=TaskMetadata(retries=1))(data=taxi_path)
    q1 = map_task(taxi_q1_task)(df=df)
    q2 = map_task(taxi_q2_task)(df=df)
    q3 = map_task(taxi_q3_task)(df=df)
    q4 = map_task(taxi_q4_task)(df=df)
    return "Ok"


if __name__ == "__main__":
    start = time.time()
    print(taxi_wf())
    print("--- %s seconds ---" % (time.time() - start))
