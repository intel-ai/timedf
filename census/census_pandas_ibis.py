# coding: utf-8
import os
import sys
import argparse
import warnings
import time
import gzip
#import mysql.connector
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
from utils import import_pandas_into_module_namespace

warnings.filterwarnings("ignore")


def compare_dataframes(ibis_df, pandas_df):

    ibis_df[0].index = pandas_df[0].index
    ibis_df[1].index = pandas_df[1].index
    if ibis_df[0].equals(pandas_df[0]) and ibis_df[1].equals(pandas_df[1]):
        print("\nTables are equal")
        return True
    else:
        print("\nTables are not equal, table1:")
        print(ibis_df[0].info())
        print("\ntable2:")
        print(pandas_df[0].info())
        pd.testing.assert_frame_equal(ibis_df[0], pandas_df[0])
        pd.testing.assert_frame_equal(ibis_df[1], pandas_df[1])
        return False


# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz
def load_data(
        filename, columns_names=None, columns_types=None, header=None, nrows=None, use_gzip=False
):
    types = None
    if columns_types:
        types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    return pd.read_csv(filename, names=columns_names, nrows=nrows, header=header, dtype=types,
                       compression='gzip' if use_gzip else None)

def etl_pandas_modified(filename, columns_names, columns_types):
    etl_times = {
        "t_readcsv": 0.0,
        "t_where": 0.0,
        "t_arithm": 0.0,
        "t_fillna": 0.0,
        "t_drop": 0.0,
        "t_typeconvert": 0.0,
        "t_etl": 0.0,
    }

    t0 = timer()
    df = load_data(filename=filename, columns_names=columns_names, columns_types=columns_types,
                   header=0, nrows=None, use_gzip=filename.endswith(".gz"))
    etl_times["t_readcsv"] = timer() - t0

    t_etl_start = timer()

    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        'CPI99',
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
    t0 = timer()
    df = df[keep_cols]
    etl_times["t_drop"] += timer() - t0

    t0 = timer()
    df = df.query("INCTOT != 9999999")
    df = df.query("EDUC != -1")
    df = df.query("EDUCD != -1")
    etl_times["t_where"] += timer() - t0

    t0 = timer()
    df["INCTOT"] = df["INCTOT"] * df["CPI99"]
    etl_times["t_arithm"] += timer() - t0

    for column in keep_cols:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times["t_fillna"] += timer() - t0

        t0 = timer()
        df[column] = df[column].astype("float64")
        etl_times["t_typeconvert"] += timer() - t0

    t0 = timer()
    y = df["EDUC"]
    X = df.drop(columns=["EDUC", "CPI99"])
    etl_times["t_drop"] += timer() - t0

    etl_times["t_etl"] = timer() - t_etl_start
    print("DataFrame shape:", X.shape)

    return X, y, etl_times

def etl_pandas(filename, columns_names, columns_types):
    etl_times = {
        "t_readcsv": 0.0,
        "t_where": 0.0,
        "t_arithm": 0.0,
        "t_fillna": 0.0,
        "t_drop": 0.0,
        "t_typeconvert": 0.0,
        "t_etl": 0.0,
    }

    t0 = timer()
    df = load_data(filename=filename, columns_names=columns_names, columns_types=columns_types,
                   header=0, nrows=None, use_gzip=filename.endswith(".gz"))
    etl_times["t_readcsv"] = timer() - t0

    t_etl_start = timer()
    df = df.query("INCTOT != 9999999")
    etl_times["t_where"] += timer() - t_etl_start

    t0 = timer()
    df["INCTOT"] = df["INCTOT"] * df["CPI99"]
    etl_times["t_arithm"] += timer() - t0

    suspect = [
        "CBSERIAL",
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
    ]
    for column in suspect:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times["t_fillna"] += timer() - t0

    totincome = ["EDUC", "EDUCD"]
    for column in totincome:
        t0 = timer()
        df = df.query(column + " != -1")
        etl_times["t_where"] += timer() - t0

    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
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
    t0 = timer()
    df = df[keep_cols]
    etl_times["t_drop"] += timer() - t0

    for column in keep_cols:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times["t_fillna"] += timer() - t0

        t0 = timer()
        df[column] = df[column].astype("float64")
        etl_times["t_typeconvert"] += timer() - t0

    y = df["EDUC"]
    t0 = timer()
    X = df.drop(columns=["EDUC"])
    etl_times["t_drop"] += timer() - t0

    etl_times["t_etl"] = timer() - t_etl_start
    print("DataFrame shape:", df.shape)

    return X, y, etl_times


def etl_ibis(
    filename,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
):

    etl_times = {
        "t_readcsv": 0.0,
        "t_where": 0.0,
        "t_arithm": 0.0,
        "t_fillna": 0.0,
        "t_pandas_drop": 0.0,
        "t_drop": 0.0,
        "t_typeconvert": 0.0,
        "t_etl": 0.0,
    }

    import ibis

    time.sleep(2)
    omnisci_server_worker.connect_to_server()

    omnisci_server_worker.create_database(
        database_name, delete_if_exists=delete_old_database
    )

    t0 = timer()

    omnisci_server_worker.connect_to_server()
    # Create table and import data
    if create_new_table:
        # Datafiles import
        t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
            table_name=table_name,
            data_files_names=filename,
            files_limit=1,
            columns_names=columns_names,
            columns_types=columns_types,
            header=0,
            nrows=None,
        )

    etl_times["t_readcsv"] = t_import_pandas + t_import_ibis

    # Second connection - this is ibis's ipc connection for DML
    conn_ipc = omnisci_server_worker.ipc_connect_to_server()
    db = conn_ipc.database(database_name)
    table = db.table(table_name)

    t_etl_start = timer()

    keep_cols = [
        'YEAR0',
        'DATANUM',
        'SERIAL',
        'CBSERIAL',
        'HHWT',
        'CPI99',
        'GQ',
        'PERNUM',
        'SEX',
        'AGE',
        'INCTOT',
        'EDUC',
        'EDUCD',
        'EDUC_HEAD',
        'EDUC_POP',
        'EDUC_MOM',
        'EDUCD_MOM2',
        'EDUCD_POP2',
        'INCTOT_MOM',
        'INCTOT_POP',
        'INCTOT_MOM2',
        'INCTOT_POP2',
        'INCTOT_HEAD',
        'SEX_HEAD',
    ]
    table = table[keep_cols]
    etl_times["t_drop"] += timer() - t_etl_start    

    # first, we do all filters and eliminate redundant fillna operations for EDUC and EDUCD
    t0 = timer()
    table = table[table.INCTOT != 9999999]
    table = table[table['EDUC'].notnull()]
    table = table[table['EDUCD'].notnull()]      
    etl_times["t_where"] += timer() - t0

    t0 = timer()
    table = table.set_column("INCTOT", table["INCTOT"] * table["CPI99"])
    etl_times["t_arithm"] += timer() - t0

  
    cols = []
    # final fillna and casting for necessary columns
    for column in keep_cols:
        t0 = timer()
        cols.append(ibis.case().when(table[column].notnull(),table[column]).else_(-1).end().cast("float64").name(column))
        etl_times["t_fillna"] += timer() - t0

    table = table.mutate(cols)

    df = table.execute()

    # here we use pandas to split table
    t0 = timer()
    y = df['EDUC']
    X = df.drop(['EDUC', 'CPI99'], axis=1)
    etl_times['t_pandas_drop'] = timer() - t0

    etl_times["t_etl"] = timer() - t_etl_start
    print("DataFrame shape:", X.shape)

    return X, y, etl_times


def print_times(etl_times, backend, db_reporter=None):
    print(f"{backend} times:")
    for time_name, time in etl_times.items():
        print("{} = {:.5f} s".format(time_name, time))
        if db_reporter is not None:
            db_reporter.submit({
                'QueryName': time_name,
                'FirstExecTimeMS': time*1000,
                'WorstExecTimeMS': time*1000,
                'BestExecTimeMS': time*1000,
                'AverageExecTimeMS': time*1000,
                'TotalTimeMS': time*1000,
                'BackEnd': backend
            })

def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def ml(X, y, random_state, n_runs, train_size, optimizer):
    if optimizer == "intel":
        print("Intel optimized sklearn is used")
        from daal4py.sklearn.model_selection import train_test_split
        import daal4py.sklearn.linear_model as lm
    if optimizer == "stock":
        print("Stock sklearn is used")
        from sklearn.model_selection import train_test_split
        import sklearn.linear_model as lm
    else:
        print(
            f"Intel optimized and stock sklearn are supported. {optimizer} can't be recognized"
        )
        sys.exit(1)

    clf = lm.Ridge()

    mse_values, cod_values = [], []
    ml_times = {"t_split":0.0, "t_ML": 0.0, "t_train": 0.0, "t_inference": 0.0}

    print("ML runs: ", n_runs)
    for i in range(n_runs):
        t0 = timer()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        ml_times["t_split"] += timer() - t0
        random_state += 777

        t0 = timer()
        model = clf.fit(X_train, y_train)
        ml_times["t_train"] += timer() - t0

        t0 = timer()
        y_pred = model.predict(X_test)
        ml_times["t_inference"] += timer() - t0

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    ml_times["t_ML"] += ml_times["t_train"] + ml_times["t_inference"]

    mse_mean = sum(mse_values) / len(mse_values)
    cod_mean = sum(cod_values) / len(cod_values)
    mse_dev = pow(
        sum([(mse_value - mse_mean) ** 2 for mse_value in mse_values])
        / (len(mse_values) - 1),
        0.5,
    )
    cod_dev = pow(
        sum([(cod_value - cod_mean) ** 2 for cod_value in cod_values])
        / (len(cod_values) - 1),
        0.5,
    )

    return mse_mean, cod_mean, mse_dev, cod_dev, ml_times

def main():
    omniscript_path = os.path.dirname(__file__)
    args = None
    omnisci_server = None

    parser = argparse.ArgumentParser(description="Run internal tests from ibis project")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-f",
        "--file",
        dest="file",
        required=True,
        help="A datafile that should be loaded",
    )
    optional.add_argument("-dnd", action="store_true", help="Do not delete old table.")
    optional.add_argument(
        "-dni",
        action="store_true",
        help="Do not create new table and import any data from CSV files.",
    )
    optional.add_argument(
        "-val",
        action="store_true",
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-o",
        "--optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default="intel",
        help="Which optimizer is used",
    )
    # MySQL database parameters
    optional.add_argument(
        "-db-server",
        dest="db_server",
        default="localhost",
        help="Host name of MySQL server.",
    )
    optional.add_argument(
        "-db-port",
        dest="db_port",
        default=3306,
        type=int,
        help="Port number of MySQL server.",
    )
    optional.add_argument(
        "-db-user",
        dest="db_user",
        default="",
        help="Username to use to connect to MySQL database. "
        "If user name is specified, script attempts to store results in MySQL "
        "database using other -db-* parameters.",
    )
    optional.add_argument(
        "-db-pass",
        dest="db_pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database.",
    )
    optional.add_argument(
        "-db-name",
        dest="db_name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results.",
    )
    optional.add_argument(
        "-db-table",
        dest="db_table",
        help="Table to use to store results for this benchmark.",
    )
    # Omnisci server parameters
    optional.add_argument(
        "-e",
        "--executable",
        dest="omnisci_executable",
        required=False,
        help="Path to omnisci_server executable.",
    )
    optional.add_argument(
        "-w",
        "--workdir",
        dest="omnisci_cwd",
        help="Path to omnisci working directory. "
        "By default parent directory of executable location is used. "
        "Data directory is used in this location.",
    )
    optional.add_argument(
        "-port",
        "--omnisci_port",
        dest="omnisci_port",
        default=6274,
        type=int,
        help="TCP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-u",
        "--user",
        dest="user",
        default="admin",
        help="User name to use on omniscidb server.",
    )
    optional.add_argument(
        "-p",
        "--password",
        dest="password",
        default="HyperInteractive",
        help="User password to use on omniscidb server.",
    )
    optional.add_argument(
        "-n",
        "--name",
        dest="name",
        default="census_database",
        help="Database name to use in omniscidb server.",
    )
    optional.add_argument(
        "-t",
        "--table",
        dest="table",
        default="census_table",
        help="Table name name to use in omniscidb server.",
    )

    optional.add_argument(
        "-commit_omnisci",
        dest="commit_omnisci",
        default="1234567890123456789012345678901234567890",
        help="Omnisci commit hash to use for benchmark.",
    )
    optional.add_argument(
        "-commit_ibis",
        dest="commit_ibis",
        default="1234567890123456789012345678901234567890",
        help="Ibis commit hash to use for benchmark.",
    )
    optional.add_argument(
        "-no_ibis",
        action="store_true",
        help="Do not run Ibis benchmark, run only Pandas (or Modin) version"
    )
    optional.add_argument(
        "-pandas_mode",
        choices=["pandas", "modin_on_ray", "modin_on_dask"],
        default="pandas",
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask"
    )
    optional.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. It should have enough space to keep -ray_memory"
    )
    optional.add_argument(
        "-ray_memory",
        default=200*1024*1024*1024,
        help="Size of memory to allocate for Ray plasma store"
    )
    optional.add_argument(
        "-no_ml",
        action="store_true",
        help="Do not run machine learning benchmark, only ETL part"
    )

    args = parser.parse_args()
    args.file = args.file.replace("'", "")

    # ML specific
    N_RUNS = 50
    TRAIN_SIZE = 0.9
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

    db_reporter = None

    try:
        if not args.no_ibis:
            if args.omnisci_executable is None:
                parser.error("Omnisci executable should be specified with -e/--executable")
            omnisci_server = OmnisciServer(
                omnisci_executable=args.omnisci_executable,
                omnisci_port=args.omnisci_port,
                database_name=args.name,
                user=args.user,
                password=args.password,
            )
            omnisci_server.launch()
            from server_worker import OmnisciServerWorker
            omnisci_server_worker = OmnisciServerWorker(omnisci_server)

            if args.db_user is not "":
                print("Connecting to database")
                db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user,
                                             passwd=args.db_pass, db=args.db_name)
                db_reporter = DbReport(db, args.db_table, {
                    'QueryName': 'VARCHAR(500) NOT NULL',
                    'FirstExecTimeMS': 'BIGINT UNSIGNED',
                    'WorstExecTimeMS': 'BIGINT UNSIGNED',
                    'BestExecTimeMS': 'BIGINT UNSIGNED',
                    'AverageExecTimeMS': 'BIGINT UNSIGNED',
                    'TotalTimeMS': 'BIGINT UNSIGNED',
                    'IbisCommitHash': 'VARCHAR(500) NOT NULL',
                    'BackEnd': 'VARCHAR(100) NOT NULL'
                }, {
                    'ScriptName': 'census_pandas_ibis.py',
                    'CommitHash': args.commit_omnisci,
                    'IbisCommitHash': args.commit_ibis
                })
                
            X_ibis, y_ibis, etl_times_ibis = etl_ibis(
                filename=args.file,
                columns_names=columns_names,
                columns_types=columns_types,
                database_name=args.name,
                table_name=args.table,
                omnisci_server_worker=omnisci_server_worker,
                delete_old_database=not args.dnd,
                create_new_table=not args.dni,
            )
            omnisci_server.terminate()
            omnisci_server = None
            print_times(etl_times_ibis, 'Ibis', db_reporter)

            if not args.no_ml:
                mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
                    X_ibis, y_ibis, RANDOM_STATE, N_RUNS, TRAIN_SIZE, args.optimizer
                )
                print_times(ml_times, 'Ibis')
                print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
                print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))

        import_pandas_into_module_namespace(main.__globals__,
                                            args.pandas_mode, args.ray_tmpdir, args.ray_memory)
        X, y, etl_times = etl_pandas_modified(
            args.file, columns_names=columns_names, columns_types=columns_types
        )
        print_times(etl_times, args.pandas_mode, db_reporter)

        if not args.no_ml:
            mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
                X, y, RANDOM_STATE, N_RUNS, TRAIN_SIZE, args.optimizer
            )
            print_times(ml_times, args.pandas_mode)
            print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
            print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))

        if args.val:
            compare_dataframes(ibis_df=(X_ibis, y_ibis), pandas_df=(X, y))
    except Exception as err:
        print("Failed: ", err)
        sys.exit(1)
    finally:
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == "__main__":
    main()
