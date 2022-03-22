# from prefect.task_runners import DaskTaskRunner, RayTaskRunner
import sys
import warnings
from collections import OrderedDict
from timeit import default_timer as timer

import pandas as pd
import xgboost

from prefect import flow, task
from prefect.deployments import DeploymentSpec

# from prefect.flow_runners import KubernetesFlowRunner, DockerFlowRunner
from prefect.flow_runners import UniversalFlowRunner

module_path = "../utils/"
if module_path not in sys.path:
    sys.path.append(module_path)


from utils import cod, mse

warnings.filterwarnings("ignore")

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

filename = "s3://modin-datasets/santander/train.csv"

ETL_KEYS = ["t_readcsv", "t_etl", "t_connect"]
ML_KEYS = ["t_train_test_split", "t_dmatrix", "t_training", "t_infer", "t_ml"]
ML_SCORE_KEYS = ["mse_mean", "cod_mean", "mse_dev"]

VAR_COLS = ["var_%s" % i for i in range(200)]
COLUMNS_NAMES = ["ID_code", "target"] + VAR_COLS
COLUMNS_TYPES = ["object", "int64"] + ["float64" for _ in range(200)]
COLUMNS_TYPES = OrderedDict((zip(COLUMNS_NAMES, COLUMNS_TYPES)))


def split_step(data, target):
    t0 = timer()
    train, valid = data[:-10000], data[-10000:]
    split_time = timer() - t0

    x_train = train.drop([target], axis=1)

    y_train = train[target]

    x_test = valid.drop([target], axis=1)

    y_test = valid[target]

    return (x_train, y_train, x_test, y_test), split_time


@task
def etl_pandas(filename, columns_names, columns_types, etl_keys):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    train_pd = pd.read_csv(
        filename, names=columns_names, dtype=columns_types, header=0, nrows=1000
    )
    etl_times["t_readcsv"] = timer() - t0

    t_etl_begin = timer()

    for i in range(200):
        col = "var_%d" % i
        var_count = train_pd.groupby(col).agg({col: "count"})

        var_count.columns = ["%s_count" % col]
        var_count = var_count.reset_index()

        train_pd = train_pd.merge(var_count, on=col, how="left")

    for i in range(200):
        col = "var_%d" % i

        mask = train_pd["%s_count" % col] > 1
        train_pd.loc[mask, "%s_gt1" % col] = train_pd.loc[mask, col]

    train_pd = train_pd.drop(["ID_code"], axis=1)
    etl_times["t_etl"] = timer() - t_etl_begin

    return train_pd, etl_times


@task
def ml(ml_data, target, ml_keys, ml_score_keys):
    ml_times = {key: 0.0 for key in ml_keys}
    ml_scores = {key: 0.0 for key in ml_score_keys}

    (x_train, y_train, x_test, y_test), ml_times["t_train_test_split"] = split_step(
        ml_data, target
    )

    t0 = timer()
    training_dmat_part = xgboost.DMatrix(data=x_train, label=y_train)
    testing_dmat_part = xgboost.DMatrix(data=x_test, label=y_test)
    ml_times["t_dmatrix"] = timer() - t0

    watchlist = [(testing_dmat_part, "eval"), (training_dmat_part, "train")]
    #     hard_code: cpu_params cannot be an input, cause values are not homogeneous
    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 1,
        "nthread": 56,
        "eta": 0.1,
        "silent": 1,
        "subsample": 0.5,
        "colsample_bytree": 0.05,
        "eval_metric": "auc",
    }

    t0 = timer()
    model = xgboost.train(
        xgb_params,
        dtrain=training_dmat_part,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=30,
        maximize=True,
        verbose_eval=1000,
    )
    ml_times["t_train"] = timer() - t0

    t0 = timer()
    yp = model.predict(testing_dmat_part)
    ml_times["t_inference"] = timer() - t0

    ml_scores["mse"] = mse(y_test, yp)
    ml_scores["cod"] = cod(y_test, yp)

    ml_times["t_ml"] += ml_times["t_train"] + ml_times["t_inference"]

    return ml_scores, ml_times


# task_runner=DaskTaskRunner()
# task_runner=RayTaskRunner()


@flow()
def santander_flow():
    df, etl_times = etl_pandas(filename, COLUMNS_NAMES, COLUMNS_TYPES, ETL_KEYS).result()
    return ml(df, "target", ML_KEYS, ML_SCORE_KEYS)


DeploymentSpec(flow=santander_flow, name="test-deployment", flow_runner=UniversalFlowRunner())
