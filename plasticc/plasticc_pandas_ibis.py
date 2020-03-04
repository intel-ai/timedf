import argparse
import sys
from functools import partial
from timeit import default_timer as timer
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

parser = argparse.ArgumentParser(description="PlasTiCC benchmark")
parser.add_argument(
    "datapath",
    metavar="datapath",
    type=str,
    help="data folder path. It should contain training_set.csv and test_set.csv",
)
parser.add_argument(
    "--gpu-memory-g",
    dest="gpu_memory",
    type=int,
    help="specify the memory of your gpu, default 16. (This controls the lines to be used. Also work for CPU version. )",
    default=16,
)
args = parser.parse_args()
print(args)

# PARAMETERS

PATH = args.datapath
GPU_MEMORY = args.gpu_memory

TEST_ROWS = 453653104
OVERHEAD = 1.2
SKIP_ROWS = int((1 - GPU_MEMORY / (32.0 * OVERHEAD)) * TEST_ROWS)

# without header
TEST_ROWS_AFTER_SKIP = 189022127

# test_set_skiprows.csv was be created via
# `head -n 189022128 test_set.csv > test_set_skiprows.csv`

t_groupby_agg = 0.0
t_arithm = 0.0
t_drop = 0.0
t_merge = 0.0
t_readcsv = 0.0
t_sort_values = 0.0
t_train_test_split = 0.0
t_dmatrix = 0.0
t_training = 0.0
t_infer = 0.0


def ravel_column_names(cols):
    d0 = cols.get_level_values(0)
    d1 = cols.get_level_values(1)
    return ["%s_%s" % (i, j) for i, j in zip(d0, d1)]


def skew_workaround(table):
    n = table['flux_count']
    m = table['flux_mean']
    s1 = table['flux_sum1']
    s2 = table['flux_sum2']
    s3 = table['flux_sum3']

    # change column name: 'skew' -> 'flux_skew'
    skew = (n * (n - 1).sqrt() / (n - 2) * (s3 - 3 * m *
                                            s2 + 2 * m * m * s1) / (s2 - m * s1).pow(1.5)).name('flux_skew')
    table = table.mutate(skew)

    return table


def etl_cpu_ibis(table, table_meta):
    global t_arithm, t_groupby_agg, t_drop, t_merge

    t0 = timer()
    table = table.mutate(flux_ratio_sq=(table['flux'] / table['flux_err']) ** 2)
    table = table.mutate(flux_by_flux_ratio_sq=table['flux'] * table['flux_ratio_sq'])
    t_arithm += timer() - t0

    aggs = [
        table.passband.mean().name("passband_mean"),
        table.flux.min().name("flux_min"),
        table.flux.max().name("flux_max"),
        table.flux.mean().name("flux_mean"),
        table.flux_err.min().name("flux_err_min"),
        table.flux_err.max().name("flux_err_max"),
        table.flux_err.mean().name("flux_err_mean"),
        table.detected.mean().name("detected_mean"),
        table.mjd.min().name("mjd_min"),
        table.mjd.max().name("mjd_max"),
        table.flux_ratio_sq.sum().name("flux_ratio_sq_sum"),
        table.flux_by_flux_ratio_sq.sum().name("flux_by_flux_ratio_sq_sum"),
        # for skew computation - should be dropped after
        table.flux.count().name("flux_count"),
        table.flux.sum().name("flux_sum1"),
        (table["flux"] ** 2).sum().name("flux_sum2"),
        (table["flux"] ** 3).sum().name("flux_sum3"),
    ]

    t0 = timer()
    table = table.groupby("object_id").aggregate(aggs)
    t_groupby_agg += timer() - t0
    
    t0 = timer()
    table = table.mutate(flux_diff=table["flux_max"] - table["flux_min"])
    table = table.mutate(flux_dif2=(table["flux_max"] - table["flux_min"]) / table["flux_mean"])
    table = table.mutate(flux_w_mean=table["flux_by_flux_ratio_sq_sum"] / table["flux_ratio_sq_sum"])
    table = table.mutate(flux_dif3=(table["flux_max"] - table["flux_min"]) / table["flux_w_mean"])
    table = table.mutate(mjd_diff=table["mjd_max"] - table["mjd_min"])

    # skew compute
    table = skew_workaround(table)

    t_arithm += timer() - t0

    t0 = timer()
    table = table.drop(["mjd_max", "mjd_min"])

    # drop temp columns using for skew computation 
    table = table.drop(['flux_count', 'flux_sum1', 'flux_sum2', 'flux_sum3'])
    t_drop += timer() - t0

    t0 = timer()

    # Problem type(table_meta) = <class 'ibis.omniscidb.client.OmniSciDBTable'>
    # which overrides the drop method (now it is used to delete the table) and
    # not for drop columns - use workaround table_meta[table_meta].drop(...)
    table_meta = table_meta[table_meta].drop(["ra", "decl", "gal_l", "gal_b"])

    t_drop += timer() - t0

    t0 = timer()
    # df_meta = df_meta.merge(agg_df, on="object_id", how="left")
    # try to workaround
    table_meta = table_meta.join(table, ["object_id"], how="left")[
        table_meta,
        table.passband_mean,
        table.flux_min,
        table.flux_max,
        table.flux_mean,
        table.flux_err_min,
        table.flux_err_max,
        table.flux_err_mean,
        table.detected_mean,
        table.flux_ratio_sq_sum,
        table.flux_by_flux_ratio_sq_sum,
        table.flux_diff,
        table.flux_dif2,
        table.flux_w_mean,
        table.flux_dif3,
        table.mjd_diff,
    ]
    t_merge += timer() - t0

    return table_meta.execute()

def etl_cpu_pandas(df, df_meta):
    global t_arithm, t_groupby_agg, t_drop, t_merge

    tl0 = timer()
    df["flux_ratio_sq"] = np.power(df["flux"] / df["flux_err"], 2.0)
    df["flux_by_flux_ratio_sq"] = df["flux"] * df["flux_ratio_sq"]
    t_arithm += timer() - tl0

    aggs = {
        "passband": ["mean"],
        "flux": ["min", "max", "mean", "skew"],
        "flux_err": ["min", "max", "mean"],
        "detected": ["mean"],
        "mjd": ["max", "min"],
        "flux_ratio_sq": ["sum"],
        "flux_by_flux_ratio_sq": ["sum"],
    }
    tl0 = timer()
    agg_df = df.groupby("object_id").agg(aggs)
    t_groupby_agg += timer() - tl0

    agg_df.columns = ravel_column_names(agg_df.columns)

    tl0 = timer()
    agg_df["flux_diff"] = agg_df["flux_max"] - agg_df["flux_min"]
    agg_df["flux_dif2"] = (agg_df["flux_max"] - agg_df["flux_min"]) / agg_df["flux_mean"]
    agg_df["flux_w_mean"] = (agg_df["flux_by_flux_ratio_sq_sum"] / agg_df["flux_ratio_sq_sum"])
    agg_df["flux_dif3"] = (agg_df["flux_max"] - agg_df["flux_min"]) / agg_df["flux_w_mean"]
    agg_df["mjd_diff"] = agg_df["mjd_max"] - agg_df["mjd_min"]
    t_arithm += timer() - tl0

    tl0 = timer()
    agg_df = agg_df.drop(["mjd_max", "mjd_min"], axis=1)
    t_drop += timer() - tl0

    agg_df = agg_df.reset_index()

    tl0 = timer()
    df_meta = df_meta.drop(["ra", "decl", "gal_l", "gal_b"], axis=1)
    t_drop += timer() - tl0

    tl0 = timer()
    df_meta = df_meta.merge(agg_df, on="object_id", how="left")
    t_merge += timer() - tl0

    return df_meta


def load_data_ibis():
    dtypes = OrderedDict({
        "object_id": "int32",
        "mjd": "float32",
        "passband": "int32",
        "flux": "float32",
        "flux_err": "float32",
        "detected": "int32",
    })

    import ibis
    print(ibis.__version__)
    conn = ibis.omniscidb.connect(
        host="localhost",
        port="6274",
        user="admin",
        password="HyperInteractive",
    )

    database_name = "plasticc_database"
    # conn.create_database(database_name)

    schema = ibis.Schema(names=dtypes.keys(), types=dtypes.values())

    # create table #1
    training_file = "%s/training_set.csv" % PATH
    try:
        conn.drop_table(
            table_name="training", database=database_name, force=True
        )
        conn.create_table(
            table_name="training", schema=schema, database=database_name
        )
    except Exception as e:
        print(e)

    # create table #2
    test_file = "%s/test_set_skiprows.csv" % PATH
    try:
        conn.drop_table(
            table_name="test", database=database_name, force=True
        )
        conn.create_table(
            table_name="test", schema=schema, database=database_name
        )
    except Exception as e:
        print(e)

    db = conn.database(database_name)
    training_table = db.table("training")
    test_table = db.table("test")

    training_table.read_csv(training_file, header=True, quoted=False, delimiter=',')
    test_table.read_csv(test_file, header=True, quoted=False, delimiter=',')


    # load metadata
    cols = [
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
    meta_dtypes = OrderedDict({cols[i]: meta_dtypes[i] for i in range(len(meta_dtypes))})

    meta_schema = ibis.Schema(names=meta_dtypes.keys(), types=meta_dtypes.values())

    # create table #3
    training_meta_file = "%s/training_set_metadata.csv" % PATH
    try:
        conn.drop_table(
            table_name="training_meta", database=database_name, force=True
        )
        conn.create_table(
            table_name="training_meta", schema=meta_schema, database=database_name
        )
    except Exception as e:
        print(e)


    del meta_dtypes["target"]
    meta_schema = ibis.Schema(names=meta_dtypes.keys(), types=meta_dtypes.values())

    # create table #4
    test_meta_file = "%s/test_set_metadata.csv" % PATH
    try:
        conn.drop_table(
            table_name="test_meta", database=database_name, force=True
        )
        conn.create_table(
            table_name="test_meta", schema=meta_schema, database=database_name
        )
    except Exception as e:
        print(e)

    training_meta_table = db.table("training_meta")
    test_meta_table = db.table("test_meta")

    training_meta_table.read_csv(training_meta_file, header=True, quoted=False, delimiter=',')
    test_meta_table.read_csv(test_meta_file, header=True, quoted=False, delimiter=',')

    return training_table, training_meta_table, test_table, test_meta_table

def load_data_pandas():
    dtypes = {
        "object_id": "int32",
        "mjd": "float32",
        "passband": "int32",
        "flux": "float32",
        "flux_err": "float32",
        "detected": "int32",
    }

    train = pd.read_csv("%s/training_set.csv" % PATH, dtype=dtypes)
    test = pd.read_csv(
        "%s/test_set_skiprows.csv" % PATH, dtype=dtypes
    )

    # load metadata
    cols = [
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
    dtypes = ["int32"] + ["float32"] * 4 + ["int32"] + ["float32"] * 5 + ["int32"]
    dtypes = {cols[i]: dtypes[i] for i in range(len(dtypes))}

    train_meta = pd.read_csv("%s/training_set_metadata.csv" % PATH, dtype=dtypes)
    del dtypes["target"]
    test_meta = pd.read_csv("%s/test_set_metadata.csv" % PATH, dtype=dtypes)

    return train, train_meta, test, test_meta


def etl_all_ibis():
    print("ibis_version")
    global t_readcsv, t_groupby_agg, t_sort_values, t_merge, t_drop, t_train_test_split
    t_etl_start = timer()

    import pdb;pdb.set_trace()

    t0 = timer()
    train, train_meta, test, test_meta = load_data_ibis()
    t_readcsv += timer() - t0

    train_final = etl_cpu_ibis(train, train_meta)
    test_final = etl_cpu_ibis(test, test_meta)

    t0 = timer()
    X = train_final.drop(["object_id", "target"], axis=1).values
    Xt = test_final.drop(["object_id"], axis=1).values
    t_drop += timer() - t0

    y = train_final["target"]
    assert X.shape[1] == Xt.shape[1]
    classes = sorted(y.unique())

    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})

    lbl = LabelEncoder()
    y = lbl.fit_transform(y)
    # print(lbl.classes_)

    t0 = timer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=126
    )
    t_train_test_split += timer() - t0

    t_etl_end = timer()

    print("t_readcsv = ", t_readcsv)
    print("t_ETL = ", t_etl_end - t_etl_start - t_readcsv)
    print("  groupby_agg = ", t_groupby_agg)
    print("  vector arithmetic = ", t_arithm)
    print("  drop columns = ", t_drop)
    print("  merge = ", t_merge)
    print("  sort_values = ", t_sort_values)
    print("  train_test_split = ", t_train_test_split)

    return X_train, y_train, X_test, y_test, Xt, classes, class_weights


def etl_all_pandas():
    print("pandas version")
    global t_readcsv, t_groupby_agg, t_sort_values, t_merge, t_drop, t_train_test_split
    t_etl_start = timer()

    t0 = timer()
    train, train_meta, test, test_meta = load_data_pandas()
    t_readcsv += timer() - t0

    train_final = etl_cpu(train, train_meta)
    test_final = etl_cpu(test, test_meta)

    t0 = timer()
    X = train_final.drop(["object_id", "target"], axis=1).values
    Xt = test_final.drop(["object_id"], axis=1).values
    t_drop += timer() - t0

    y = train_final["target"]
    assert X.shape[1] == Xt.shape[1]
    classes = sorted(y.unique())

    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})

    lbl = LabelEncoder()
    y = lbl.fit_transform(y)
    # print(lbl.classes_)

    t0 = timer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=126
    )
    t_train_test_split += timer() - t0

    t_etl_end = timer()

    print("t_readcsv = ", t_readcsv)
    print("t_ETL = ", t_etl_end - t_etl_start - t_readcsv)
    print("  groupby_agg = ", t_groupby_agg)
    print("  vector arithmetic = ", t_arithm)
    print("  drop columns = ", t_drop)
    print("  merge = ", t_merge)
    print("  sort_values = ", t_sort_values)
    print("  train_test_split = ", t_train_test_split)

    return X_train, y_train, X_test, y_test, Xt, classes, class_weights


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
    loss = multi_weighted_logloss(
        y_true.get_label(), y_predicted, classes, class_weights
    )
    return "wloss", loss


def ml(X_train, y_train, X_test, y_test, Xt, classes, class_weights):
    global t_dmatrix, t_training, t_infer

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
        xgb_multi_weighted_logloss, classes=classes, class_weights=class_weights
    )

    t_ml_start = timer()
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalid = xgb.DMatrix(data=X_test, label=y_test)
    dtest = xgb.DMatrix(data=Xt)
    t_dmatrix += timer() - t_ml_start

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
    t_training += timer() - t0

    t0 = timer()
    yp = clf.predict(dvalid)
    t_infer += timer() - t0

    cpu_loss = multi_weighted_logloss(y_test, yp, classes, class_weights)

    t0 = timer()
    ysub = clf.predict(dtest)
    t_infer += timer() - t0

    t_ml_end = timer()

    print("t_ML:", t_ml_end - t_ml_start)
    print("  t_dmatrix = ", t_dmatrix)
    print("  t_train =", t_training)
    print("  t_pred = ", t_infer)

    print("validation cpu_loss:", cpu_loss)


def main():
    X_train, y_train, X_test, y_test, Xt, classes, class_weights = etl_all_ibis()

    ml(X_train, y_train, X_test, y_test, Xt, classes, class_weights)


if __name__ == "__main__":
    main()
