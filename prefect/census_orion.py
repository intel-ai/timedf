# from prefect.task_runners import DaskTaskRunner, RayTaskRunner
import sys
import warnings
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import config_context

from prefect import flow, task
from prefect.deployments import DeploymentSpec

# from prefect.flow_runners import KubernetesFlowRunner, DockerFlowRunner
from prefect.flow_runners import UniversalFlowRunner

module_path = "../utils/"
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import cod, mse, split

warnings.filterwarnings("ignore")


# dataset = "https://flyte-datasets.s3.us-east-2.amazonaws.com/census.csv"
dataset = "s3://modin-datasets/census/ipums_education2income_1970-2010.csv"
ML_KEYS = ["t_train_test_split", "t_train", "t_inference", "t_ml"]
ML_SCORE_KEYS = ["mse_mean", "cod_mean", "mse_dev"]

N_RUNS = 50
TEST_SIZE = 0.1
RANDOM_STATE = 777

COLS = [
    "YEAR",
    "DATANUM",
    "SERIAL",
    "CBSERIAL",
    "HHWT",
    "CPI99",
    "GQ",
    "PERNUM",
    "SEX",
    "AGE",
    "INCTOT",
    "EDUC",
    "EDUCD",
    "EDUC_HEAD",
    "EDUC_POP",
    "EDUC_MOM",
    "EDUCD_MOM2",
    "EDUCD_POP2",
    "INCTOT_MOM",
    "INCTOT_POP",
    "INCTOT_MOM2",
    "INCTOT_POP2",
    "INCTOT_HEAD",
    "SEX_HEAD",
]

COLUMNS_TYPES = [
    "int",
    "int",
    "int",
    "float",
    "int",
    "float",
    "int",
    "float",
    "int",
    "int",
    "int",
    "int",
    "int",
    "int",
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
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
    "float",
]


@task
def feature_eng_task(data, cols):

    df = pd.read_csv(data, compression="infer", nrows=1000)[cols]

    df = df[df["INCTOT"] != 9999999]
    df = df[df["EDUC"] != -1]
    df = df[df["EDUCD"] != -1]

    df["INCTOT"] = df["INCTOT"] * df["CPI99"]

    for column in cols:
        df[column] = df[column].fillna(-1)
        df[column] = df[column].astype("float64")

    return df


@task
def ml_task(df, random_state, n_runs, test_size, ml_keys, ml_score_keys):

    # Fetch the input and output data from train dataset
    y = np.ascontiguousarray(df["EDUC"], dtype=np.float64)
    X = np.ascontiguousarray(df.drop(columns=["EDUC", "CPI99"]), dtype=np.float64)

    clf = lm.Ridge()

    mse_values, cod_values = [], []
    ml_times = {key: 0.0 for key in ml_keys}
    ml_scores = {key: 0.0 for key in ml_score_keys}

    print("ML runs: ", n_runs)
    for i in range(n_runs):
        (X_train, y_train, X_test, y_test), split_time = split(
            X, y, test_size=test_size, random_state=random_state
        )
        ml_times["t_train_test_split"] += split_time
        random_state += 777

        t0 = timer()
        with config_context(assume_finite=True):
            model = clf.fit(X_train, y_train)
        ml_times["t_train"] += timer() - t0

        t0 = timer()
        y_pred = model.predict(X_test)
        ml_times["t_inference"] += timer() - t0

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    ml_times["t_ml"] += ml_times["t_train"] + ml_times["t_inference"]

    ml_scores["mse_mean"] = sum(mse_values) / len(mse_values)
    ml_scores["cod_mean"] = sum(cod_values) / len(cod_values)
    ml_scores["mse_dev"] = pow(
        sum([(mse_value - ml_scores["mse_mean"]) ** 2 for mse_value in mse_values])
        / (len(mse_values) - 1),
        0.5,
    )
    ml_scores["cod_dev"] = pow(
        sum([(cod_value - ml_scores["cod_mean"]) ** 2 for cod_value in cod_values])
        / (len(cod_values) - 1),
        0.5,
    )

    return ml_scores, ml_times


@flow()
def census_flow():
    df = feature_eng_task(dataset, COLS)
    ml_scores, ml_times = ml_task(
        df, RANDOM_STATE, N_RUNS, TEST_SIZE, ML_KEYS, ML_SCORE_KEYS
    ).result()


DeploymentSpec(flow=census_flow, name="test-deployment", flow_runner=UniversalFlowRunner())
