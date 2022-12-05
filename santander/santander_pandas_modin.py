# coding: utf-8
import warnings
from decimal import Decimal
from timeit import default_timer as timer

from utils import (
    check_support,
    cod,
    import_pandas_into_module_namespace,
    load_data_pandas,
    load_data_modin_on_hdk,
    mse,
    print_results,
)

warnings.filterwarnings("ignore")

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data


def etl(filename, columns_names, columns_types, etl_keys, pandas_mode):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    if pandas_mode == "Modin_on_hdk":
        train_pd = load_data_modin_on_hdk(
            filename=filename,
            columns_names=columns_names,
            columns_types=columns_types,
            skiprows=1,
            pd=run_benchmark.__globals__["pd"],
        )
    else:
        train_pd = load_data_pandas(
            filename=filename,
            columns_names=columns_names,
            columns_types=columns_types,
            header=0,
            nrows=None,
            use_gzip=filename.endswith(".gz"),
            pd=run_benchmark.__globals__["pd"],
        )
    etl_times["t_readcsv"] = timer() - t0

    t_etl_begin = timer()

    for i in range(1):
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


def split_step(data, target):
    t0 = timer()
    train, valid = data[:-10000], data[-10000:]
    split_time = timer() - t0

    x_train = train.drop([target], axis=1)

    y_train = train[target]

    x_test = valid.drop([target], axis=1)

    y_test = valid[target]

    return (x_train, y_train, x_test, y_test), split_time


def ml(ml_data, target, ml_keys, ml_score_keys):
    import xgboost

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


def run_benchmark(parameters):
    check_support(parameters, unsupported_params=["dfiles_num", "gpu_memory", "optimizer"])

    parameters["data_file"] = parameters["data_file"].replace("'", "")
    parameters["no_ml"] = parameters["no_ml"] or False

    var_cols = ["var_%s" % i for i in range(200)]
    columns_names = ["ID_code", "target"] + var_cols
    columns_types_pd = ["object", "int64"] + ["float64" for _ in range(200)]

    etl_keys = ["t_readcsv", "t_etl", "t_connect"]
    ml_keys = ["t_train_test_split", "t_ml", "t_train", "t_inference", "t_dmatrix"]
    ml_score_keys = ["mse", "cod"]

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )

    ml_data, etl_times = etl(
        filename=parameters["data_file"],
        columns_names=columns_names,
        columns_types=columns_types_pd,
        etl_keys=etl_keys,
        pandas_mode=parameters["pandas_mode"],
    )
    print_results(results=etl_times, backend=parameters["pandas_mode"], unit="s")
    etl_times["Backend"] = parameters["pandas_mode"]

    results = {"ETL": [etl_times]}
    if not parameters["no_ml"]:
        ml_scores, ml_times = ml(
            ml_data=ml_data, target="target", ml_keys=ml_keys, ml_score_keys=ml_score_keys
        )
        print_results(results=ml_times, backend=parameters["pandas_mode"], unit="s")
        ml_times["Backend"] = parameters["pandas_mode"]
        print_results(results=ml_scores, backend=parameters["pandas_mode"])
        ml_scores["Backend"] = parameters["pandas_mode"]
        results["ML"] = [ml_scores]

    return results
