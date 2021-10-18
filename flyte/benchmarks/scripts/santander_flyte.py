# coding: utf-8
import time

# from collections import OrderedDict
# from dataclasses import dataclass
import typing
import warnings
from timeit import default_timer as timer

# from flytekit import Resources
# from flytekit.types.file import FlyteFile
# from flytekit.types.schema import FlyteSchema
import pandas as pd
import xgboost
from flytekit import task, workflow

warnings.filterwarnings("ignore")

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

ETL_KEYS = ["t_readcsv", "t_etl", "t_connect"]
ML_KEYS = ["t_train_test_split", "t_dmatrix", "t_training", "t_infer", "t_ml"]
ML_SCORE_KEYS = ["mse_mean", "cod_mean", "mse_dev"]

VAR_COLS = ["var_%s" % i for i in range(200)]
COLUMNS_NAMES = ["ID_code", "target"] + VAR_COLS
COLUMNS_TYPES = ["object", "int64"] + ["float64" for _ in range(200)]


def load_data_pandas(
    filename,
    columns_names=None,
    columns_types=None,
    header=None,
    nrows=None,
    use_gzip=False,
    parse_dates=None,
):
    types = None
    if columns_types:
        types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    return pd.read_csv(
        filename,
        names=columns_names,
        nrows=nrows,
        header=header,
        dtype=types,
        compression="gzip" if use_gzip else None,
        parse_dates=parse_dates,
    )


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


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
def etl_pandas(
    filename: str,
    columns_names: typing.List[str],
    columns_types: typing.List[str],
    etl_keys: typing.List[str],
) -> (pd.DataFrame, typing.Dict[str, float]):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    train_pd = load_data_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        header=0,
        use_gzip=filename.endswith(".gz"),
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
def ml(
    ml_data: pd.DataFrame, target: str, ml_keys: typing.List[str], ml_score_keys: typing.List[str]
) -> (typing.Dict[str, float], typing.Dict[str, float]):
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


@workflow
def santander_ml_wf(
    filename: str = "/santander_train.csv",  # full path is to write
    columns_names: typing.List[str] = COLUMNS_NAMES,
    columns_types: typing.List[str] = COLUMNS_TYPES,
    etl_keys: typing.List[str] = ETL_KEYS,
    target: str = "target",
    ml_keys: typing.List[str] = ML_KEYS,
    ml_score_keys: typing.List[str] = ML_SCORE_KEYS,
) -> (typing.Dict[str, float], typing.Dict[str, float]):
    df, etl_times = etl_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        etl_keys=etl_keys,
    )
    return ml(ml_data=df, target=target, ml_keys=ml_keys, ml_score_keys=ml_score_keys)


if __name__ == "__main__":
    start = time.time()
    print(santander_ml_wf())
    print("--- %s seconds ---" % (time.time() - start))
