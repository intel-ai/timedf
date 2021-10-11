from collections import OrderedDict
from functools import partial
from timeit import default_timer as timer

import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder

from utils import (
    check_fragments_size,
    check_support,
    compare_dataframes,
    import_pandas_into_module_namespace,
    print_results,
    split,
)


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


def etl_cpu_ibis(table, table_meta, etl_times):
    t_etl_start = timer()

    table = table.mutate(flux_ratio_sq=(table["flux"] / table["flux_err"]) ** 2)
    table = table.mutate(flux_by_flux_ratio_sq=table["flux"] * table["flux_ratio_sq"])

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

    table = table.groupby("object_id").aggregate(aggs)
    table = table.mutate(flux_diff=table["flux_max"] - table["flux_min"])
    table = table.mutate(flux_dif2=table["flux_diff"] / table["flux_mean"])
    table = table.mutate(
        flux_w_mean=table["flux_by_flux_ratio_sq_sum"] / table["flux_ratio_sq_sum"]
    )
    table = table.mutate(flux_dif3=table["flux_diff"] / table["flux_w_mean"])
    table = table.mutate(mjd_diff=table["mjd_max"] - table["mjd_min"])
    # skew compute
    table = skew_workaround(table)

    table = table.drop(["mjd_max", "mjd_min", "flux_count", "flux_sum1", "flux_sum2", "flux_sum3"])

    table_meta = table_meta.drop(["ra", "decl", "gal_l", "gal_b"])

    # df_meta = df_meta.merge(agg_df, on="object_id", how="left")
    # try to workaround
    table_meta = table_meta.join(table, ["object_id"], how="left")[
        table_meta,
        table.passband_mean,
        table.flux_min,
        table.flux_max,
        table.flux_mean,
        table.flux_skew,
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

    df = table_meta.execute()

    etl_times["t_etl"] += timer() - t_etl_start

    return df


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


def load_data_ibis(
    dataset_path,
    database_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    skip_rows,
    validation,
    dtypes,
    meta_dtypes,
    import_mode,
    fragments_size,
):
    fragments_size = check_fragments_size(
        fragments_size,
        count_table=4,
        import_mode=import_mode,
        default_fragments_size=[32000000, 32000000, 32000000, 32000000],
    )

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)

    t_readcsv = 0.0
    t_connect = 0.0

    # Create tables and import data
    if create_new_table:
        import ibis

        training_file = "%s/training_set.csv" % dataset_path
        # COPY FROM doesn't have skip_rows option
        test_file = "%s/test_set_skiprows.csv" % dataset_path
        training_meta_file = "%s/training_set_metadata.csv" % dataset_path
        test_meta_file = "%s/test_set_metadata.csv" % dataset_path

        schema = ibis.Schema(names=dtypes.keys(), types=dtypes.values())
        meta_schema = ibis.Schema(names=meta_dtypes.keys(), types=meta_dtypes.values())

        target = meta_dtypes.pop("target")
        meta_schema_without_target = ibis.Schema(
            names=meta_dtypes.keys(), types=meta_dtypes.values()
        )
        meta_dtypes["target"] = target

        if import_mode == "copy-from":
            # create tables
            t0 = timer()
            omnisci_server_worker.create_table(
                table_name="training",
                schema=schema,
                database=database_name,
                fragment_size=fragments_size[0],
            )
            omnisci_server_worker.create_table(
                table_name="test",
                schema=schema,
                database=database_name,
                fragment_size=fragments_size[1],
            )
            omnisci_server_worker.create_table(
                table_name="training_meta",
                schema=meta_schema,
                database=database_name,
                fragment_size=fragments_size[2],
            )
            omnisci_server_worker.create_table(
                table_name="test_meta",
                schema=meta_schema_without_target,
                database=database_name,
                fragment_size=fragments_size[3],
            )

            # get tables
            db = omnisci_server_worker.database(database_name)
            training_table = db.table("training")
            test_table = db.table("test")
            training_meta_table = db.table("training_meta")
            test_meta_table = db.table("test_meta")
            t_connect = timer() - t0

            # measuring time of reading
            t0 = timer()
            training_table.read_csv(training_file, header=True, quotechar="", delimiter=",")
            test_table.read_csv(test_file, header=True, quotechar="", delimiter=",")
            training_meta_table.read_csv(
                training_meta_file, header=True, quotechar="", delimiter=","
            )
            test_meta_table.read_csv(test_meta_file, header=True, quotechar="", delimiter=",")
            t_readcsv = timer() - t0

        elif import_mode == "pandas":
            general_options = {
                "files_limit": 1,
                "columns_names": list(dtypes.keys()),
                "columns_types": list(dtypes.values()),
                "header": 0,
                "nrows": None,
                "compression_type": None,
                "validation": validation,
            }

            # create table #1
            t_import_pandas_1, t_import_ibis_1 = omnisci_server_worker.import_data_by_ibis(
                table_name="training",
                data_files_names="%s/training_set.csv" % dataset_path,
                **general_options,
            )

            # create table #2
            t_import_pandas_2, t_import_ibis_2 = omnisci_server_worker.import_data_by_ibis(
                table_name="test",
                data_files_names="%s/test_set.csv" % dataset_path,
                skiprows=skip_rows,
                **general_options,
            )

            # before reading meta test files, we should update columns_names
            # and columns_types in general options with its proper values
            general_options["columns_names"] = list(meta_dtypes.keys())
            general_options["columns_types"] = list(meta_dtypes.values())

            # create table #3
            t_import_pandas_3, t_import_ibis_3 = omnisci_server_worker.import_data_by_ibis(
                table_name="training_meta",
                data_files_names="%s/training_set_metadata.csv" % dataset_path,
                **general_options,
            )

            target = meta_dtypes.pop("target")
            general_options["columns_names"] = list(meta_dtypes.keys())
            general_options["columns_types"] = list(meta_dtypes.values())

            # create table #4
            t_import_pandas_4, t_import_ibis_4 = omnisci_server_worker.import_data_by_ibis(
                table_name="test_meta",
                data_files_names="%s/test_set_metadata.csv" % dataset_path,
                **general_options,
            )
            meta_dtypes["target"] = target

            t_import_pandas = (
                t_import_pandas_1 + t_import_pandas_2 + t_import_pandas_3 + t_import_pandas_4
            )
            t_import_ibis = t_import_ibis_1 + t_import_ibis_2 + t_import_ibis_3 + t_import_ibis_4
            print(f"import times: pandas - {t_import_pandas}s, ibis - {t_import_ibis}s")
            t_readcsv = t_import_pandas + t_import_ibis
            t_connect += omnisci_server_worker.get_conn_creation_time()

        elif import_mode == "fsi":
            t0 = timer()
            omnisci_server_worker._conn.create_table_from_csv(
                "training",
                training_file,
                schema,
                fragment_size=fragments_size[0],
            )
            omnisci_server_worker._conn.create_table_from_csv(
                "test",
                test_file,
                schema,
                fragment_size=fragments_size[1],
            )
            omnisci_server_worker._conn.create_table_from_csv(
                "training_meta",
                training_meta_file,
                meta_schema,
                fragment_size=fragments_size[2],
            )
            omnisci_server_worker._conn.create_table_from_csv(
                "test_meta",
                test_meta_file,
                meta_schema_without_target,
                fragment_size=fragments_size[3],
            )
            t_readcsv = timer() - t0
            t_connect += omnisci_server_worker.get_conn_creation_time()

    # Second connection - this is ibis's ipc connection for DML
    t0 = timer()
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    db = omnisci_server_worker.database(database_name)
    t_connect += timer() - t0

    training_table = db.table("training")
    test_table = db.table("test")

    training_meta_table = db.table("training_meta")
    test_meta_table = db.table("test_meta")

    return (
        training_table,
        training_meta_table,
        test_table,
        test_meta_table,
        t_readcsv,
        t_connect,
    )


def load_data_pandas(dataset_path, skip_rows, dtypes, meta_dtypes, pandas_mode):
    # 'pd' module is defined implicitly in 'import_pandas_into_module_namespace'
    # function so we should use 'noqa: F821' for flake8
    train = pd.read_csv("%s/training_set.csv" % dataset_path, dtype=dtypes)  # noqa: F821
    # Currently we need to avoid skip_rows in Mode_on_omnisci mode since
    # pyarrow uses it in incompatible way
    if pandas_mode == "Modin_on_omnisci":
        test = pd.read_csv(  # noqa: F821
            "%s/test_set_skiprows.csv" % dataset_path,
            names=list(dtypes.keys()),
            dtype=dtypes,
            header=0,
        )
    else:
        test = pd.read_csv(  # noqa: F821
            "%s/test_set.csv" % dataset_path,
            names=list(dtypes.keys()),
            dtype=dtypes,
            skiprows=skip_rows,
        )

    train_meta = pd.read_csv(  # noqa: F821
        "%s/training_set_metadata.csv" % dataset_path, dtype=meta_dtypes
    )
    target = meta_dtypes.pop("target")
    test_meta = pd.read_csv(  # noqa: F821
        "%s/test_set_metadata.csv" % dataset_path, dtype=meta_dtypes
    )
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


def etl_all_ibis(
    dataset_path,
    database_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    skip_rows,
    validation,
    dtypes,
    meta_dtypes,
    etl_keys,
    import_mode,
    fragments_size,
):
    print("Ibis version")
    etl_times = {key: 0.0 for key in etl_keys}

    print("importing data ...")
    (
        train,
        train_meta,
        test,
        test_meta,
        etl_times["t_readcsv"],
        etl_times["t_connect"],
    ) = load_data_ibis(
        dataset_path=dataset_path,
        database_name=database_name,
        omnisci_server_worker=omnisci_server_worker,
        delete_old_database=delete_old_database,
        create_new_table=create_new_table,
        ipc_connection=ipc_connection,
        skip_rows=skip_rows,
        validation=validation,
        dtypes=dtypes,
        meta_dtypes=meta_dtypes,
        import_mode=import_mode,
        fragments_size=fragments_size,
    )

    # update etl_times
    print("preprocessing train data ...")
    train_final = etl_cpu_ibis(train, train_meta, etl_times)
    print("preprocessing test data ...")
    test_final = etl_cpu_ibis(test, test_meta, etl_times)

    return train_final, test_final, etl_times


def etl_all_pandas(dataset_path, skip_rows, dtypes, meta_dtypes, etl_keys, pandas_mode):
    print("Pandas version")
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
    train_final = etl_cpu_pandas(train, train_meta, etl_times)
    test_final = etl_cpu_pandas(test, test_meta, etl_times)

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

    # if you want to use ibis' read_csv then you need to manually create
    # test_set_skiprows.csv (for example, via next command:
    # `head -n 189022128 test_set.csv > test_set_skiprows.csv`)
    #
    # for gpu_memory=16 - skip_rows=189022127 (+1 for header)

    overhead = 1.2
    skip_rows = int((1 - gpu_memory / (32.0 * overhead)) * test_rows)
    return skip_rows


def run_benchmark(parameters):
    check_support(parameters, unsupported_params=["dfiles_num"])

    parameters["data_file"] = parameters["data_file"].replace("'", "")
    parameters["gpu_memory"] = parameters["gpu_memory"] or 16
    parameters["no_ml"] = parameters["no_ml"] or False

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

    if not parameters["no_pandas"]:
        import_pandas_into_module_namespace(
            namespace=run_benchmark.__globals__,
            mode=parameters["pandas_mode"],
            ray_tmpdir=parameters["ray_tmpdir"],
            ray_memory=parameters["ray_memory"],
        )

    etl_times_ibis = None
    ml_times_ibis = None
    etl_times = None
    ml_times = None

    if not parameters["no_ibis"]:
        train_final_ibis, test_final_ibis, etl_times_ibis = etl_all_ibis(
            dataset_path=parameters["data_file"],
            database_name=parameters["database_name"],
            omnisci_server_worker=parameters["omnisci_server_worker"],
            delete_old_database=not parameters["dnd"],
            create_new_table=not parameters["dni"],
            ipc_connection=parameters["ipc_connection"],
            skip_rows=skip_rows,
            validation=parameters["validation"],
            dtypes=dtypes,
            meta_dtypes=meta_dtypes,
            etl_keys=etl_keys,
            import_mode=parameters["import_mode"],
            fragments_size=parameters["fragments_size"],
        )

        print_results(results=etl_times_ibis, backend="Ibis", unit="s")
        etl_times_ibis["Backend"] = "Ibis"

        if not parameters["no_ml"]:
            print("using ml with dataframes from Ibis")
            ml_times_ibis = ml(train_final_ibis, test_final_ibis, ml_keys)
            print_results(results=ml_times_ibis, backend="Ibis", unit="s")
            ml_times_ibis["Backend"] = "Ibis"

    if not parameters["no_pandas"]:
        train_final, test_final, etl_times = etl_all_pandas(
            dataset_path=parameters["data_file"],
            skip_rows=skip_rows,
            dtypes=dtypes,
            meta_dtypes=meta_dtypes,
            etl_keys=etl_keys,
            pandas_mode=parameters["pandas_mode"],
        )

        print_results(results=etl_times, backend=parameters["pandas_mode"], unit="s")
        etl_times["Backend"] = parameters["pandas_mode"]

        if not parameters["no_ml"]:
            print("using ml with dataframes from Pandas")
            ml_times = ml(
                train_final, test_final, ml_keys, use_modin_xgb=parameters["use_modin_xgb"]
            )
            print_results(results=ml_times, backend=parameters["pandas_mode"], unit="s")
            ml_times["Backend"] = parameters["pandas_mode"]
            ml_times["Backend"] = (
                parameters["pandas_mode"]
                if not parameters["use_modin_xgb"]
                else parameters["pandas_mode"] + "_modin_xgb"
            )

    if parameters["validation"] and parameters["import_mode"] != "pandas":
        print(
            "WARNING: validation can not be performed, it works only for 'pandas' import mode, '{}' passed".format(
                parameters["import_mode"]
            )
        )

    if parameters["validation"] and parameters["import_mode"] == "pandas":
        compare_dataframes(
            ibis_dfs=[train_final_ibis, test_final_ibis],
            pandas_dfs=[train_final, test_final],
            parallel_execution=True,
        )

    return {"ETL": [etl_times_ibis, etl_times], "ML": [ml_times_ibis, ml_times]}
