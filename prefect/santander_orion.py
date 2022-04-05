import sys
import warnings

import pandas as pd

from prefect import flow, task
from prefect.deployments import DeploymentSpec

# from prefect.flow_runners import KubernetesFlowRunner, DockerFlowRunner
from prefect.flow_runners import UniversalFlowRunner

module_path = "../santander/"
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = "../utils/"
if module_path not in sys.path:
    sys.path.append(module_path)

from santander_pandas_modin import *
from utils import cod, mse

warnings.filterwarnings("ignore")

run_benchmark.__globals__["pd"] = pd

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

filename = "s3://modin-datasets/santander/train.csv"

ETL_KEYS = ["t_readcsv", "t_etl", "t_connect"]
ML_KEYS = ["t_train_test_split", "t_dmatrix", "t_training", "t_infer", "t_ml"]
ML_SCORE_KEYS = ["mse_mean", "cod_mean", "mse_dev"]

VAR_COLS = ["var_%s" % i for i in range(200)]
COLUMNS_NAMES = ["ID_code", "target"] + VAR_COLS
COLUMNS_TYPES = ["object", "int64"] + ["float64" for _ in range(200)]


etl_task = task(etl)
ml_task = task(ml)


@flow()
def santander_flow():
    df, etl_times = etl_task(filename, COLUMNS_NAMES, COLUMNS_TYPES, ETL_KEYS).result()
    return ml_task(df, "target", ML_KEYS, ML_SCORE_KEYS)


DeploymentSpec(flow=santander_flow, name="santander", flow_runner=UniversalFlowRunner())
