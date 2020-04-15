# coding: utf-8
import os
import sys
import time
import traceback
import warnings
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    check_fragments_size,
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
    split,
)

warnings.filterwarnings("ignore")


# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz


def etl_pandas(filename, columns_names, columns_types, etl_keys):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = timer()
    df = load_data_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        header=0,
        nrows=None,
        use_gzip=filename.endswith(".gz"),
        pd=run_benchmark.__globals__["pd"],
    )
    etl_times["t_readcsv"] = round((timer() - t0) * 1000)

    t_etl_start = timer()

    keep_cols = [
        "YEAR0",
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
    df = df[keep_cols]

    df = df.query("INCTOT != 9999999")
    df = df.query("EDUC != -1")
    df = df.query("EDUCD != -1")

    df["INCTOT"] = df["INCTOT"] * df["CPI99"]

    for column in keep_cols:
        df[column] = df[column].fillna(-1)

        df[column] = df[column].astype("float64")

    y = df["EDUC"]
    X = df.drop(columns=["EDUC", "CPI99"])

    etl_times["t_etl"] = round((timer() - t_etl_start) * 1000)
    print("DataFrame shape:", X.shape)

    return df, X, y, etl_times


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

    fragments_size = check_fragments_size(
        fragments_size, count_table=1, import_mode=import_mode, default_fragments_size=[1000000]
    )

    etl_times = {key: 0.0 for key in etl_keys}

    omnisci_server_worker.create_database(database_name, delete_if_exists=delete_old_database)

    # Create table and import data
    if create_new_table:
        schema_table = ibis.Schema(names=columns_names, types=columns_types)
        if import_mode == "copy-from":
            # Create table and import data for ETL queries
            omnisci_server_worker.create_table(
                table_name=table_name,
                schema=schema_table,
                database=database_name,
                fragment_size=fragments_size[0],
            )
            table_import = omnisci_server_worker.database(database_name).table(table_name)

            t0 = timer()
            table_import.read_csv(filename, header=True, quotechar="", delimiter=",")
            etl_times["t_readcsv"] = round((timer() - t0) * 1000)

        elif import_mode == "pandas":
            # Datafiles import
            t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
                table_name=table_name,
                data_files_names=filename,
                files_limit=1,
                columns_names=columns_names,
                columns_types=columns_types,
                header=0,
                nrows=None,
                compression_type="gzip" if filename.endswith("gz") else None,
                validation=validation,
            )
            etl_times["t_readcsv"] = round((t_import_pandas + t_import_ibis) * 1000)

        elif import_mode == "fsi":
            try:
                unzip_name = None
                if filename.endswith("gz"):
                    import gzip

                    unzip_name = "/tmp/census-fsi.csv"

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
                etl_times["t_readcsv"] = round((timer() - t0) * 1000)

            finally:
                if filename.endswith("gz"):
                    import os

                    os.remove(unzip_name)

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.connect_to_server(database_name, ipc=ipc_connection)
    table = omnisci_server_worker.database(database_name).table(table_name)

    t_etl_start = timer()

    keep_cols = [
        "YEAR0",
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

    if import_mode == "pandas" and validation:
        keep_cols.append("id")

    table = table[keep_cols]

    # first, we do all filters and eliminate redundant fillna operations for EDUC and EDUCD
    table = table[table.INCTOT != 9999999]
    table = table[table["EDUC"].notnull()]
    table = table[table["EDUCD"].notnull()]

    table = table.set_column("INCTOT", table["INCTOT"] * table["CPI99"])

    cols = []
    # final fillna and casting for necessary columns
    for column in keep_cols:
        cols.append(
            ibis.case()
            .when(table[column].notnull(), table[column])
            .else_(-1)
            .end()
            .cast("float64")
            .name(column)
        )

    table = table.mutate(cols)

    df = table.execute()

    if import_mode == "pandas" and validation:
        df.index = df["id"].values

    # here we use pandas to split table
    y = df["EDUC"]
    X = df.drop(["EDUC", "CPI99"], axis=1)

    etl_times["t_etl"] = round((timer() - t_etl_start) * 1000)
    print("DataFrame shape:", X.shape)

    return df, X, y, etl_times


def ml(X, y, random_state, n_runs, test_size, optimizer, ml_keys, ml_score_keys):
    if optimizer == "intel":
        print("Intel optimized sklearn is used")
        import daal4py.sklearn.linear_model as lm
    elif optimizer == "stock":
        print("Stock sklearn is used")
        import sklearn.linear_model as lm
    else:
        print(f"Intel optimized and stock sklearn are supported. {optimizer} can't be recognized")
        sys.exit(1)

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
        model = clf.fit(X_train, y_train)
        ml_times["t_train"] += round((timer() - t0) * 1000)

        t0 = timer()
        y_pred = model.predict(X_test)
        ml_times["t_inference"] += round((timer() - t0) * 1000)

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


def run_benchmark(parameters):

    ignored_parameters = {
        "dfiles_num": parameters["dfiles_num"],
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    # ML specific
    N_RUNS = 50
    TEST_SIZE = 0.1
    RANDOM_STATE = 777

    columns_names = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "QGQ",
        "PERNUM",
        "PERWT",
        "SEX",
        "AGE",
        "EDUC",
        "EDUCD",
        "INCTOT",
        "SEX_HEAD",
        "SEX_MOM",
        "SEX_POP",
        "SEX_SP",
        "SEX_MOM2",
        "SEX_POP2",
        "AGE_HEAD",
        "AGE_MOM",
        "AGE_POP",
        "AGE_SP",
        "AGE_MOM2",
        "AGE_POP2",
        "EDUC_HEAD",
        "EDUC_MOM",
        "EDUC_POP",
        "EDUC_SP",
        "EDUC_MOM2",
        "EDUC_POP2",
        "EDUCD_HEAD",
        "EDUCD_MOM",
        "EDUCD_POP",
        "EDUCD_SP",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_HEAD",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_SP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
    ]
    columns_types = [
        "int64",
        "int64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]
    etl_keys = ["t_readcsv", "t_etl"]
    ml_keys = ["t_train_test_split", "t_ml", "t_train", "t_inference"]

    ml_score_keys = ["mse_mean", "cod_mean", "mse_dev", "cod_dev"]

    try:
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

        if parameters["validation"] and parameters["import_mode"] != "pandas":
            print(
                "WARNING: validation can not be performed, it works only for 'pandas' import mode, '{}' passed".format(
                    parameters["import_mode"]
                )
            )

        if not parameters["no_ibis"]:
            df_ibis, X_ibis, y_ibis, etl_times_ibis = etl_ibis(
                filename=parameters["data_file"],
                columns_names=columns_names,
                columns_types=columns_types,
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

            if not parameters["no_ml"]:
                ml_scores_ibis, ml_times_ibis = ml(
                    X=X_ibis,
                    y=y_ibis,
                    random_state=RANDOM_STATE,
                    n_runs=N_RUNS,
                    test_size=TEST_SIZE,
                    optimizer=parameters["optimizer"],
                    ml_keys=ml_keys,
                    ml_score_keys=ml_score_keys,
                )
                print_results(results=ml_times_ibis, backend="Ibis", unit="s")
                ml_times_ibis["Backend"] = "Ibis"
                print_results(results=ml_scores_ibis, backend="Ibis")
                ml_scores_ibis["Backend"] = "Ibis"

        if not parameters["no_pandas"]:
            df, X, y, etl_times = etl_pandas(
                parameters["data_file"],
                columns_names=columns_names,
                columns_types=columns_types,
                etl_keys=etl_keys,
            )

            print_results(results=etl_times, backend=parameters["pandas_mode"], unit="s")
            etl_times["Backend"] = parameters["pandas_mode"]

            if not parameters["no_ml"]:
                ml_scores, ml_times = ml(
                    X=X,
                    y=y,
                    random_state=RANDOM_STATE,
                    n_runs=N_RUNS,
                    test_size=TEST_SIZE,
                    optimizer=parameters["optimizer"],
                    ml_keys=ml_keys,
                    ml_score_keys=ml_score_keys,
                )
                print_results(results=ml_times, backend=parameters["pandas_mode"], unit="s")
                ml_times["Backend"] = parameters["pandas_mode"]
                print_results(results=ml_scores, backend=parameters["pandas_mode"])
                ml_scores["Backend"] = parameters["pandas_mode"]

        if parameters["validation"] and parameters["import_mode"] == "pandas":
            # this should work only for pandas mode
            compare_dataframes(ibis_dfs=(X_ibis, y_ibis), pandas_dfs=(X, y))

        return {"ETL": [etl_times_ibis, etl_times], "ML": [ml_times_ibis, ml_times]}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
