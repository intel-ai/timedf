import time

import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass
import typing
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
import numpy as np


cols = ["trip_id",
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
        "dropoff_puma"]

parse_dates=["pickup_datetime", "dropoff_datetime"]


#@task(cache_version="1.0", cache=True, limits=Resources(cpu="100m", mem="2Gi"))
@task
def get_taxi_dataset_task(
    data: FlyteFile[typing.TypeVar("csv")],
    compression: str,
    names: typing.List[str],
    parse_dates: typing.List[str]
) -> typing.NamedTuple("OutputsBC", get_taxi_output=pd.DataFrame):
    return pd.read_csv(data, compression=compression, names=cols, parse_dates=parse_dates)


#@task(cache_version="1.0", cache=True, limits=Resources(cpu="100m", mem="2Gi"))
@task
def taxi_q1_task(
    df: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(df.groupby(["cab_type"]).count()["trip_id"])


#@task(cache_version="1.0", cache=True, limits=Resources(cpu="100m", mem="2Gi"))
@task
def taxi_q2_task(
    df: pd.DataFrame
) -> pd.DataFrame:
    return df.groupby("passenger_count", as_index=False).mean()[["passenger_count", "total_amount"]]


@task
def taxi_q3_task(
    df: pd.DataFrame
) -> pd.DataFrame:
    res = df.groupby(["passenger_count", "pickup_datetime"]).size().reset_index()
    res.columns = res.columns.astype(str)
    return res


@task
def taxi_q4_task(
    df: pd.DataFrame
) -> pd.DataFrame:
    transformed = pd.DataFrame({
        "passenger_count": df["passenger_count"],
        "pickup_datetime": df["pickup_datetime"].dt.year,
        "trip_distance": df["trip_distance"].astype("int64"),
    })
    transformed = transformed.groupby(["passenger_count", "pickup_datetime", "trip_distance"])  \
            .size().reset_index().sort_values(by=["pickup_datetime", 0], ascending=[True, False])
    transformed.columns = transformed.columns.astype(str)
    return transformed


@workflow
def taxi_wf(
    dataset: FlyteFile["csv"] = "/localdisk/taxi_data/trips_xaa.csv.gz",
    compression: str = 'gzip'
) -> pd.DataFrame:
    df = get_taxi_dataset_task(data=dataset, compression=compression, names=cols, parse_dates=parse_dates)[0]
    #res = taxi_q1_task(df=df)
    #res = taxi_q2_task(df=df)
    #res = taxi_q3_task(df=df)
    res = taxi_q4_task(df=df)
    return res


if __name__ == "__main__":
    start = time.time()
    print(taxi_wf())
    print("--- %s seconds ---" % (time.time() - start))
