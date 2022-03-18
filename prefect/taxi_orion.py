import pandas as pd

from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import UniversalFlowRunner

# from modin import pandas as pd


# from prefect.flow_runners import DockerFlowRunner, KubernetesFlowRunner


# datapath = "https://modin-datasets.s3.amazonaws.com/taxi/trips_xaa_5M.csv.gz"
datapath = "https://flyte-datasets.s3.us-east-2.amazonaws.com/trips_head100.csv"

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

parse_dates = ["pickup_datetime", "dropoff_datetime"]


@task
def get_taxi_dataset_task(datapath, names, parse_dates):
    return pd.read_csv(datapath, names=names, parse_dates=parse_dates)


@task
def taxi_q1_task(df):
    return pd.DataFrame(df.groupby(["cab_type"]).count()["trip_id"])


@task
def taxi_q2_task(df):
    return df.groupby("passenger_count", as_index=False).mean()[
        ["passenger_count", "total_amount"]
    ]


@task
def taxi_q3_task(df):
    res = df.groupby(["passenger_count", "pickup_datetime"]).size().reset_index()
    res.columns = res.columns.astype(str)
    return res


@task
def taxi_q4_task(df):
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


@flow
def taxi_queries_flow():

    df = get_taxi_dataset_task(datapath, cols, parse_dates)
    q1 = taxi_q1_task(df)
    q2 = taxi_q2_task(df)
    q3 = taxi_q3_task(df)
    q4 = taxi_q4_task(df)
    return q1, q2, q3, q4


DeploymentSpec(name="taxi", flow=taxi_queries_flow, flow_runner=UniversalFlowRunner())
