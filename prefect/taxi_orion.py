import sys

from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import UniversalFlowRunner

# from prefect.flow_runners import DockerFlowRunner, KubernetesFlowRunner

module_path = "../taxi/"
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = "../utils/"
if module_path not in sys.path:
    sys.path.append(module_path)

from taxibench_pandas_modin import q1, q2, q3, q4, run_benchmark
from utils import load_data_pandas

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
    return load_data_pandas(
        filename=datapath,
        columns_names=names,
        header=None,
        nrows=None,
        use_gzip=datapath.endswith(".gz"),
        parse_dates=parse_dates,
        pd=run_benchmark.__globals__["pd"],
        pandas_mode="Pandas",
    )


@task
def taxi_q1_task(df, pandas_mode="Pandas"):
    q1(df, pandas_mode)


@task
def taxi_q2_task(df, pandas_mode="Pandas"):
    q2(df, pandas_mode)


@task
def taxi_q3_task(df, pandas_mode="Pandas"):
    q3(df, pandas_mode)


@task
def taxi_q4_task(df, pandas_mode="Pandas"):
    q4(df, pandas_mode)


@flow
def taxi_queries_flow():
    df = get_taxi_dataset_task(datapath, cols, parse_dates)
    q1 = taxi_q1_task(df)
    q2 = taxi_q2_task(df)
    q3 = taxi_q3_task(df)
    q4 = taxi_q4_task(df)
    return q1, q2, q3, q4


DeploymentSpec(name="taxi", flow=taxi_queries_flow, flow_runner=UniversalFlowRunner())
