import argparse
from collections import OrderedDict
from functools import partial
from timeit import default_timer as timer

import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder

from timedf import BaseBenchmark, BenchmarkResults
from timedf.pandas_backend import pd
from timedf.benchmark_utils import print_results, split


def ravel_column_names(cols):
    d0 = cols.get_level_values(0)
    d1 = cols.get_level_values(1)
    return ["%s_%s" % (i, j) for i, j in zip(d0, d1)]


def skew_workaround(table):
    n = table["flux_count"]
    m = table["flux_mean"]
    s1 = table["flux_sum1"]
    s2 = table["flux_sum2"]
    s3 = table["flux_sum3"]

    # change column name: 'skew' -> 'flux_skew'
    skew = (
        n * (n - 1).sqrt() / (n - 2) * (s3 - 3 * m * s2 + 2 * m * m * s1) / (s2 - m * s1).pow(1.5)
    ).name("flux_skew")
    table = table.mutate(skew)

    return table


def etl_cpu(df, df_meta, etl_times):
    t_etl_start = timer()

    # workaround for both Modin_on_ray and Modin_on_hdk modes. Eventually this should be fixed
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


def load_data_pandas(dataset_path, skip_rows, dtypes, meta_dtypes, pandas_mode):
    train = pd.read_csv("%s/training_set.csv" % dataset_path, dtype=dtypes)
    # Currently we need to avoid skip_rows in Mode_on_hdk mode since
    # pyarrow uses it in incompatible way
    if pandas_mode == "Modin_on_hdk":
        test = pd.read_csv(
            # This file didn't come from kaggle competition
            "%s/test_set_skiprows.csv" % dataset_path,
            names=list(dtypes.keys()),
            dtype=dtypes,
            header=0,
        )
    else:
        test = pd.read_csv(
            "%s/test_set.csv" % dataset_path,
            names=list(dtypes.keys()),
            dtype=dtypes,
            skiprows=skip_rows,
        )

    train_meta = pd.read_csv("%s/training_set_metadata.csv" % dataset_path, dtype=meta_dtypes)
    target = meta_dtypes.pop("target")
    test_meta = pd.read_csv("%s/test_set_metadata.csv" % dataset_path, dtype=meta_dtypes)
    meta_dtypes["target"] = target

    return train, train_meta, test, test_meta


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


def etl(dataset_path, skip_rows, dtypes, meta_dtypes, etl_keys, pandas_mode):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    train, train_meta, test, test_meta = load_data_pandas(
        dataset_path=dataset_path,
        skip_rows=skip_rows,
        dtypes=dtypes,
        meta_dtypes=meta_dtypes,
        pandas_mode=pandas_mode,
    )
    etl_times["t_readcsv"] += timer() - t0

    # update etl_times
    train_final = etl_cpu(train, train_meta, etl_times)
    test_final = etl_cpu(test, test_meta, etl_times)

    return train_final, test_final, etl_times


def multi_weighted_logloss(y_true, y_preds, classes, class_weights, use_modin_xgb=False):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order="F")
    y_ohe = pandas.get_dummies(y_true)

    if use_modin_xgb:
        missed_columns = set(range(len(classes))) - set(np.unique(y_true))
        y_missed = pandas.DataFrame(
            np.zeros((len(y_ohe), len(missed_columns))), columns=missed_columns, index=y_ohe.index
        )
        y_ohe = pandas.concat([y_ohe, y_missed], axis=1)
        y_ohe.sort_index(axis=1, inplace=True)

    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = -np.sum(y_w) / np.sum(class_arr)
    return loss


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights, use_modin_xgb=False):
    loss = multi_weighted_logloss(
        y_true.get_label(), y_predicted, classes, class_weights, use_modin_xgb=use_modin_xgb
    )
    return "wloss", loss


def ml(train_final, test_final, ml_keys, use_modin_xgb=False):
    ml_times = {key: 0.0 for key in ml_keys}

    (
        (X_train, y_train, X_test, y_test, Xt, classes, class_weights),
        ml_times["t_train_test_split"],
    ) = split_step(train_final, test_final)

    if use_modin_xgb:
        import modin.experimental.xgboost as xgb
        import modin.pandas as pd

        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        Xt = pd.DataFrame(Xt)
    else:
        import xgboost as xgb
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

    func_loss = partial(
        xgb_multi_weighted_logloss,
        classes=classes,
        class_weights=class_weights,
        use_modin_xgb=use_modin_xgb,
    )

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

    if use_modin_xgb:
        y_test = y_test.values
        yp = yp.values

    cpu_loss = multi_weighted_logloss(y_test, yp, classes, class_weights)

    t0 = timer()
    ysub = clf.predict(dtest)  # noqa: F841 (unused variable)
    ml_times["t_infer"] += timer() - t0

    ml_times["t_ml"] = timer() - t_ml_start

    print("validation cpu_loss:", cpu_loss)

    return ml_times


def compute_skip_rows(gpu_memory):
    # count rows inside test_set.csv
    test_rows = 453653104

    overhead = 1.2
    skip_rows = int((1 - gpu_memory / (32.0 * overhead)) * test_rows)
    return skip_rows


def run_benchmark(parameters):
    skip_rows = compute_skip_rows(parameters["gpu_memory"])

    dtypes = OrderedDict(
        [
            ("object_id", "int32"),
            ("mjd", "float32"),
            ("passband", "int32"),
            ("flux", "float32"),
            ("flux_err", "float32"),
            ("detected", "int32"),
        ]
    )

    # load metadata
    columns_names = [
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
    meta_dtypes = ["int32"] + ["float32"] * 4 + ["int32"] + ["float32"] * 5 + ["int32"]
    meta_dtypes = OrderedDict(
        [(columns_names[i], meta_dtypes[i]) for i in range(len(meta_dtypes))]
    )

    etl_keys = ["t_readcsv", "t_etl", "t_connect"]
    ml_keys = ["t_train_test_split", "t_dmatrix", "t_training", "t_infer", "t_ml"]

    train_final, test_final, results = etl(
        dataset_path=parameters["data_file"],
        skip_rows=skip_rows,
        dtypes=dtypes,
        meta_dtypes=meta_dtypes,
        etl_keys=etl_keys,
        pandas_mode=parameters["pandas_mode"],
    )

    print_results(results=results, backend=parameters["pandas_mode"])

    if not parameters["no_ml"]:
        print("using ml with dataframes from Pandas")
        ml_times = ml(train_final, test_final, ml_keys, use_modin_xgb=parameters["use_modin_xgb"])
        print_results(results=ml_times, backend=parameters["pandas_mode"])
        results.update(ml_times)

    return BenchmarkResults(results)


class Benchmark(BaseBenchmark):
    __params__ = ("gpu_memory",)

    def add_benchmark_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-gpu_memory",
            type=int,
            help="specify the memory of your gpu"
            "(This controls the lines to be used. Also work for CPU version. )",
            default=16,
        )

    def run_benchmark(self, params) -> BenchmarkResults:
        return run_benchmark(params)

    def load_data(self, target_dir, reload=False):
        from timedf.tools.s3_load import download_folder

        download_folder(
            "modin-datasets", "plasticc", target_dir, reload=reload, pattern=r".*\.csv"
        )
