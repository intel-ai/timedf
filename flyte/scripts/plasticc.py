import time

from collections import OrderedDict
from functools import partial
from timeit import default_timer as timer
from dataclasses import dataclass
import typing
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.schema import FlyteSchema
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")


ETL_KEYS = ["t_readcsv", "t_etl", "t_connect"]
ML_KEYS = ["t_train_test_split", "t_dmatrix", "t_training", "t_infer", "t_ml"]


COLUMNS_NAMES = [
    "object_id",
    "ra",
    "decl",
    "gal_l",
    "gal_b",
    "ddf",
    "hostgal_specz",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv",
    "target",
]


DTYPES = OrderedDict(
    [
        ("object_id", 'int'),
        ("mjd", 'float'),
        ("passband", 'int'),
        ("flux", 'float'),
        ("flux_err", 'float'),
        ("detected", 'int'),
    ]
)

# functions from utils and external in terms of Flytekit workflow

def split(X, y, test_size=0.1, stratify=None, random_state=None):
    t0 = timer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    split_time = timer() - t0

    return (X_train, y_train, X_test, y_test), split_time


def ravel_column_names(cols):
    d0 = cols.get_level_values(0)
    d1 = cols.get_level_values(1)
    return ["%s_%s" % (i, j) for i, j in zip(d0, d1)]


def load_data_pandas(dataset_path, dtypes, meta_dtypes):
    train = pd.read_csv("%s/training_set.csv" % dataset_path, dtype=dtypes)
    
    test = pd.read_csv(
        "%s/test_set_skiprows.csv" % dataset_path
    )

    train_meta = pd.read_csv(
        "%s/training_set_metadata.csv" % dataset_path, dtype=meta_dtypes
    )
    target = meta_dtypes.pop("target")
    test_meta = pd.read_csv(
        "%s/test_set_metadata.csv" % dataset_path, dtype=meta_dtypes
    )
    meta_dtypes["target"] = target

    return train, train_meta, test, test_meta


def etl_cpu_pandas(df, df_meta, etl_times):
    t_etl_start = timer()

    # workaround for both Modin_on_ray and Modin_on_omnisci modes. Eventually this should be fixed
    df["flux_ratio_sq"] = (df["flux"] / df["flux_err"]) * (
        df["flux"] / df["flux_err"]
    )  # np.power(df["flux"] / df["flux_err"], 2.0)
    df["flux_by_flux_ratio_sq"] = df["flux"] * df["flux_ratio_sq"]

    aggs = {
        "passband": ["mean"],
        "flux": ["min", "max", "mean", "skew"],
        "flux_err": ["min", "max", "mean"],
        "detected": ["mean"],
        "mjd": ["max", "min"],
        "flux_ratio_sq": ["sum"],
        "flux_by_flux_ratio_sq": ["sum"],
    }
    agg_df = df.groupby("object_id", sort=False).agg(aggs)

    agg_df.columns = ravel_column_names(agg_df.columns)

    agg_df["flux_diff"] = agg_df["flux_max"] - agg_df["flux_min"]
    agg_df["flux_dif2"] = agg_df["flux_diff"] / agg_df["flux_mean"]
    agg_df["flux_w_mean"] = agg_df["flux_by_flux_ratio_sq_sum"] / agg_df["flux_ratio_sq_sum"]
    agg_df["flux_dif3"] = agg_df["flux_diff"] / agg_df["flux_w_mean"]
    agg_df["mjd_diff"] = agg_df["mjd_max"] - agg_df["mjd_min"]

    agg_df = agg_df.drop(["mjd_max", "mjd_min"], axis=1)

    agg_df = agg_df.reset_index()

    df_meta = df_meta.drop(["ra", "decl", "gal_l", "gal_b"], axis=1)

    df_meta = df_meta.merge(agg_df, on="object_id", how="left")

    _ = df_meta.shape
    etl_times["t_etl"] += timer() - t_etl_start

    return df_meta


def split_step(train_final, test_final):

    X = train_final.drop(["object_id", "target"], axis=1).values
    Xt = test_final.drop(["object_id"], axis=1).values

    y = train_final["target"]
    assert X.shape[1] == Xt.shape[1]
    classes = sorted(y.unique())

    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})

    lbl = LabelEncoder()
    y = lbl.fit_transform(y)

    (X_train, y_train, X_test, y_test), split_time = split(
        X, y, test_size=0.1, stratify=y, random_state=126
    )

    return (X_train, y_train, X_test, y_test, Xt, classes, class_weights), split_time


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order="F")
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = -np.sum(y_w) / np.sum(class_arr)
    return loss


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights):
    loss = multi_weighted_logloss(y_true.get_label(), y_predicted, classes, class_weights)
    return "wloss", loss


@task
def etl_all_pandas(
    dataset_path: str,
    columns_names: typing.List[str],
    dtypes: typing.Dict[str, str],    # types OrderedDict, pandas.Series, class 'type' are not supported!!!
    #meta_dtypes: typing.Dict[str, type],
    etl_keys: typing.List[str]
) -> (
    pd.DataFrame,
    pd.DataFrame,
    typing.Dict[str, float]
):
    dtypes = dict(zip(dtypes.keys(), list(map(eval, dtypes.values()))))

    meta_dtypes = [int] + [float] * 4 + [int] + [float] * 5 + [int]
    meta_dtypes = OrderedDict(
        [(columns_names[i], meta_dtypes[i]) for i in range(len(meta_dtypes))]
    )
    
    
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    train, train_meta, test, test_meta = load_data_pandas(
        dataset_path=dataset_path,
        dtypes=dtypes,
        meta_dtypes=meta_dtypes
    )
    etl_times["t_readcsv"] += timer() - t0

    # update etl_times
    train_final = etl_cpu_pandas(train, train_meta, etl_times)
    test_final = etl_cpu_pandas(test, test_meta, etl_times)

    return train_final, test_final, etl_times



@task
def ml(
    train_final: pd.DataFrame,
    test_final: pd.DataFrame,
    ml_keys: typing.List[str]
) -> typing.Dict[str, float]:
    ml_times = {key: 0.0 for key in ml_keys}

    (
        (X_train, y_train, X_test, y_test, Xt, classes, class_weights),
        ml_times["t_train_test_split"],
    ) = split_step(train_final, test_final)

#     hard_code: cpu_params cannot be an input, cause values are not homogeneous
    cpu_params = {
        "objective": "multi:softprob",
        "tree_method": "hist",
        "nthread": 16,
        "num_class": 14,
        "max_depth": 7,
        "silent": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    }

    func_loss = partial(xgb_multi_weighted_logloss, classes=classes, class_weights=class_weights)

    t_ml_start = timer()
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalid = xgb.DMatrix(data=X_test, label=y_test)
    dtest = xgb.DMatrix(data=Xt)
    ml_times["t_dmatrix"] += timer() - t_ml_start

    watchlist = [(dvalid, "eval"), (dtrain, "train")]

    t0 = timer()
    clf = xgb.train(
        cpu_params,
        dtrain=dtrain,
        num_boost_round=60,
        evals=watchlist,
        feval=func_loss,
        early_stopping_rounds=10,
        verbose_eval=1000,
    )
    ml_times["t_training"] += timer() - t0

    t0 = timer()
    yp = clf.predict(dvalid)
    ml_times["t_infer"] += timer() - t0

    cpu_loss = multi_weighted_logloss(y_test, yp, classes, class_weights)

    t0 = timer()
    ysub = clf.predict(dtest)  # noqa: F841 (unused variable)
    ml_times["t_infer"] += timer() - t0

    ml_times["t_ml"] = timer() - t_ml_start

    print("validation cpu_loss:", cpu_loss)

    return ml_times


@workflow
def plasticc_ml_wf(
    dataset_path: str = 'plasticc/',    #path to file directory
    columns_names: typing.List[str] = COLUMNS_NAMES,
    dtypes: typing.Dict[str, str] = DTYPES, 
    etl_keys: typing.List[str] = ETL_KEYS
) -> typing.Dict[str, float]:
    train_final, test_final, etl_times = etl_all_pandas(
        dataset_path=dataset_path, columns_names=columns_names, dtypes=dtypes, etl_keys=ETL_KEYS)
    return ml(train_final=train_final, test_final=test_final, ml_keys=ML_KEYS)


if __name__ == "__main__":
    start = time.time()
    print(plasticc_ml_wf())
    print("--- %s seconds ---" % (time.time() - start))
