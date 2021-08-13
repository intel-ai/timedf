import time

import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass
import typing
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

from timeit import default_timer as timer
from sklearn import config_context
import warnings

warnings.filterwarnings("ignore")


# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz

ML_KEYS = ["t_train_test_split", "t_train", "t_inference", "t_ml"]
ML_SCORE_KEYS = ["mse_mean", "cod_mean", "mse_dev"]

N_RUNS = 50
TEST_SIZE = 0.1
RANDOM_STATE = 777

# Functions from utils

def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def split(X, y, test_size=0.1, stratify=None, random_state=None):
    t0 = timer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    split_time = timer() - t0

    return (X_train, y_train, X_test, y_test), split_time

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

#X = OrderedDict((zip(cols, column_types)))
#Y = OrderedDict({"EDUC": X.pop("EDUC")})


@task
def feature_eng_task(
    data: FlyteFile[typing.TypeVar("csv")],
    cols: typing.List[str]
) -> (
    pd.DataFrame
):

    df =  pd.read_csv(data)[cols]

    df = df[df["INCTOT"] != 9999999]
    df = df[df["EDUC"] != -1]
    df = df[df["EDUCD"] != -1]

    df["INCTOT"] = df["INCTOT"] * df["CPI99"]

    for column in cols:
        df[column] = df[column].fillna(-1)
        df[column] = df[column].astype("float64")

    return df


@task
def ml_task(
    df: pd.DataFrame,
    random_state: int,
    n_runs: int,
    test_size: float,
    ml_keys: typing.List[str],
    ml_score_keys: typing.List[str]
) -> (
    typing.Dict[str, float],
    typing.Dict[str, float]
):

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
            X, y, test_size=test_size, random_state=random_state)
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


@workflow
def census_bench_wf(
    dataset: FlyteFile["csv"] = "census.csv",
    cols: typing.List[str] = COLS,
    random_state: int = RANDOM_STATE,
    n_runs: int = N_RUNS,
    test_size: float = TEST_SIZE,
    ml_keys: typing.List[str] = ML_KEYS,
    ml_score_keys: typing.List[str] = ML_SCORE_KEYS
) -> (
    typing.Dict[str, float],
    typing.Dict[str, float]
):
    df = feature_eng_task(data=dataset, cols=cols)
    ml_scores, ml_times = ml_task(df=df, random_state=random_state, n_runs=n_runs, test_size=test_size,
                                  ml_keys=ml_keys, ml_score_keys=ml_score_keys)
    return ml_scores, ml_times


if __name__ == "__main__":
    start = time.time()
    print(census_bench_wf())
    print("--- %s seconds ---" % (time.time() - start))
