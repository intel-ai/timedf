# coding: utf-8
import warnings
from timeit import default_timer as timer

from utils import (
    check_fragments_size,
    check_support,
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
    get_tmp_filepath,
)

warnings.filterwarnings("ignore")

# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

# Current script prerequisites:
# 1) Patched OmniSci version (https://github.com/intel-go/omniscidb/tree/develop)
# 2) Patched Ibis version (https://github.com/intel-go/ibis/tree/develop)


def etl_pandas(filename, columns_names, columns_types, etl_keys):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
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


def etl_ibis(
    filename,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    ipc_connection,
    validation,
    etl_keys,
    import_mode,
    fragments_size,
):
    import ibis

    etl_times = {key: 0.0 for key in etl_keys}

    fragments_size = check_fragments_size(fragments_size, count_table=1, import_mode=import_mode)

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)

    if create_new_table:
        # Create table and import data for ETL queries
        schema_table = ibis.Schema(names=columns_names, types=columns_types)
        if import_mode == "copy-from":
            t0 = timer()
            omnisci_server_worker.create_table(
                table_name=table_name,
                schema=schema_table,
                database=database_name,
                fragment_size=fragments_size[0],
            )
            table_import = omnisci_server_worker.database(database_name).table(table_name)
            etl_times["t_connect"] += timer() - t0

            t0 = timer()
            table_import.read_csv(filename, header=True, quotechar="", delimiter=",")
            etl_times["t_readcsv"] = timer() - t0

        elif import_mode == "pandas":
            # decimal(8, 4) is converted to decimal(9, 6) in order to provide better data conversion
            # accuracy during import from Pandas into OmniSciDB for proper results validation
            columns_types = [
                "decimal(9, 6)" if (x == "decimal(8, 4)") else x for x in columns_types
            ]
            t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
                table_name=table_name,
                data_files_names=filename,
                files_limit=1,
                columns_names=columns_names,
                columns_types=columns_types,
                header=0,
                nrows=None,
                compression_type="gzip" if filename.endswith(".gz") else None,
                use_columns_types_for_pd=False,
            )
            etl_times["t_readcsv"] = t_import_pandas + t_import_ibis
            etl_times["t_connect"] += omnisci_server_worker.get_conn_creation_time()

        elif import_mode == "fsi":
            try:
                unzip_name = None
                if filename.endswith(".gz"):
                    import gzip

                    unzip_name = get_tmp_filepath("santander-fsi.csv")

                    with gzip.open(filename, "rb") as gz_input:
                        with open(unzip_name, "wb") as output:
                            output.write(gz_input.read())

                t0 = timer()
                omnisci_server_worker._conn.create_table_from_csv(
                    table_name,
                    unzip_name or filename,
                    schema_table,
                    fragment_size=fragments_size[0],
                )
                etl_times["t_readcsv"] = timer() - t0
                etl_times["t_connect"] += omnisci_server_worker.get_conn_creation_time()

            finally:
                if filename.endswith("gz"):
                    import os

                    os.remove(unzip_name)

    # Second connection - this is ibis's ipc connection for DML
    t0 = timer()
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    table = omnisci_server_worker.database(database_name).table(table_name)
    etl_times["t_connect"] += timer() - t0

    # group_by/count, merge (join) and filtration queries
    # We are making 400 columns and then insert them into original table thus avoiding
    # nested sql requests
    t_etl_start = timer()
    count_cols = []
    orig_cols = ["ID_code", "target"] + ["var_%s" % i for i in range(200)]
    cast_cols = []
    cast_cols.append(table["target"].cast("int64").name("target0"))
    gt1_cols = []
    for i in range(200):
        col = "var_%d" % i
        col_count = "var_%d_count" % i
        col_gt1 = "var_%d_gt1" % i
        w = ibis.window(group_by=col)
        count_cols.append(table[col].count().over(w).name(col_count))
        gt1_cols.append(
            ibis.case()
            .when(table[col].count().over(w).name(col_count) > 1, table[col].cast("float32"))
            .else_(ibis.null())
            .end()
            .name(col_gt1)
        )
        cast_cols.append(table[col].cast("float32").name(col))

    table = table.mutate(count_cols)
    table = table.drop(orig_cols)
    table = table.mutate(gt1_cols)
    table = table.mutate(cast_cols)

    table_df = table.execute()

    etl_times["t_etl"] = timer() - t_etl_start
    return table_df, etl_times


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

    etl_times_ibis = None
    etl_times = None
    ml_times_ibis = None
    ml_times = None

    var_cols = ["var_%s" % i for i in range(200)]
    columns_names = ["ID_code", "target"] + var_cols
    columns_types_pd = ["object", "int64"] + ["float64" for _ in range(200)]
    columns_types_ibis = ["string", "int32"] + ["decimal(8, 4)" for _ in range(200)]

    etl_keys = ["t_readcsv", "t_etl", "t_connect"]
    ml_keys = ["t_train_test_split", "t_ml", "t_train", "t_inference", "t_dmatrix"]
    ml_score_keys = ["mse", "cod"]

    if not parameters["no_pandas"]:
        import_pandas_into_module_namespace(
            namespace=run_benchmark.__globals__,
            mode=parameters["pandas_mode"],
            ray_tmpdir=parameters["ray_tmpdir"],
            ray_memory=parameters["ray_memory"],
        )

    if not parameters["no_ibis"]:
        ml_data_ibis, etl_times_ibis = etl_ibis(
            filename=parameters["data_file"],
            columns_names=columns_names,
            columns_types=columns_types_ibis,
            database_name=parameters["database_name"],
            table_name=parameters["table"],
            omnisci_server_worker=parameters["omnisci_server_worker"],
            delete_old_database=not parameters["dnd"],
            create_new_table=not parameters["dni"],
            ipc_connection=parameters["ipc_connection"],
            validation=parameters["validation"],
            etl_keys=etl_keys,
            import_mode=parameters["import_mode"],
            fragments_size=parameters["fragments_size"],
        )

        print_results(results=etl_times_ibis, backend="Ibis", unit="s")
        etl_times_ibis["Backend"] = "Ibis"

    if not parameters["no_pandas"]:
        ml_data, etl_times = etl_pandas(
            filename=parameters["data_file"],
            columns_names=columns_names,
            columns_types=columns_types_pd,
            etl_keys=etl_keys,
        )
        print_results(results=etl_times, backend=parameters["pandas_mode"], unit="s")
        etl_times["Backend"] = parameters["pandas_mode"]

    if not parameters["no_ml"]:
        if not parameters["no_pandas"]:
            ml_scores, ml_times = ml(
                ml_data=ml_data, target="target", ml_keys=ml_keys, ml_score_keys=ml_score_keys
            )
            print_results(results=ml_times, backend=parameters["pandas_mode"], unit="s")
            ml_times["Backend"] = parameters["pandas_mode"]
            print_results(results=ml_scores, backend=parameters["pandas_mode"])
            ml_scores["Backend"] = parameters["pandas_mode"]

        if not parameters["no_ibis"]:
            ml_scores_ibis, ml_times_ibis = ml(
                ml_data=ml_data_ibis,
                target="target0",
                ml_keys=ml_keys,
                ml_score_keys=ml_score_keys,
            )
            print_results(results=ml_times_ibis, backend="Ibis", unit="s")
            ml_times_ibis["Backend"] = "Ibis"
            print_results(results=ml_scores_ibis, backend="Ibis")
            ml_scores_ibis["Backend"] = "Ibis"

    # Results validation block (comparison of etl_ibis and etl_pandas outputs)
    if parameters["validation"]:
        print("Validation of ETL query results ...")
        cols_to_sort = ["var_0", "var_1", "var_2", "var_3", "var_4"]

        ml_data_ibis = ml_data_ibis.rename(columns={"target0": "target"})
        # compare_dataframes doesn't sort pandas dataframes
        ml_data.sort_values(by=cols_to_sort, inplace=True)

        compare_dataframes(
            ibis_dfs=[ml_data_ibis],
            pandas_dfs=[ml_data],
            sort_cols=cols_to_sort,
            drop_cols=[],
            parallel_execution=True,
        )

    return {"ETL": [etl_times_ibis, etl_times], "ML": [ml_times_ibis, ml_times]}
