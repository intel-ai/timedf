import pandas as pd

from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import UniversalFlowRunner
# from prefect.flow_runners import DockerFlowRunner, KubernetesFlowRunner

import sys
module_path = "../taxi/"
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = "../utils/"
if module_path not in sys.path:
    sys.path.append(module_path)

from taxibench_pandas_modin import *
from utils import *
from s3_client import *


# datapath = "s3://modin-datasets/taxi/trips_xa{a}.csv"
# datapath = "s3://modin-datasets/taxi/trips_{xaa_5M}.csv.gz"
datapath = "s3://flyte-datasets/trips_{head100}.csv"

parameters = {
        "data_file": datapath,
        "dfiles_num": 1,
        "pandas_mode": "Pandas",
        "ray_tmpdir": None,
        "ray_memory": None,
        "validation": None
}


@flow
def taxi_queries_flow():
    run_benchmark(parameters)


DeploymentSpec(name="taxi", flow=taxi_queries_flow, flow_runner=UniversalFlowRunner())
