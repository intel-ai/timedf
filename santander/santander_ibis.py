#import mysql.connector
import argparse
import time
import sys
import os
import ibis
import xgboost
import pandas as pd
#import numpy as np
import pathlib

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
from server_worker import OmnisciServerWorker
from utils import execute_process

def compare_tables(table1, table2):
    
    if table1.equals(table2):
        return True
    else:
        print("\ntables are not equal, table1:")
        print(table1.info())
        print("\ntable2:")
        print(table2.info())
        return False

def import_ibis():
    
    t1, t2 = omnisci_server_worker.import_data_by_ibis(table_name=tmp_table_name,
                                                     data_files_names=args.dp, files_limit=1,
                                                     columns_names=datafile_columns_names,
                                                     columns_types=datafile_columns_types,
                                                     cast_dict=None, header=0)
    t_import = t1 + t2
    print("t_import Ibis", t_import)
    return t_import

def import_FSI():
    omnisci_server_worker.execute_sql_query(drop_tmp_table_sql_query)
    
    t0 = time.time()
    omnisci_server_worker.execute_sql_query(import_csv_by_FSI_sql_query)
    t_import = time.time() - t0
    print("t_import FSI", t_import)

    omnisci_server_worker.execute_sql_query(drop_tmp_table_sql_query)
    return t_import

def import_COPY():
    omnisci_server_worker.execute_sql_query(drop_tmp_table_sql_query)
    omnisci_server_worker.execute_sql_query(create_table_sql_query)
    
    t0 = time.time()
    omnisci_server_worker.execute_sql_query(import_csv_by_COPY_sql_query)
    t_import = time.time() - t0
    print("t_import COPY", t_import)

    omnisci_server_worker.execute_sql_query(drop_tmp_table_sql_query)
    return t_import

def etl():
    global frame
    global training_part
    global validation_part
    cur_table = df
    
    t0 = time.time()

    # We are making 400 columns and then insert them into original table thus avoiding nested sql requests
    count_cols = []
    orig_cols = ["ID_code"]
    cast_cols = []    
    for i in range(200):
        col = 'var_%d' % i
        col_count = 'var_%d_count' % i
        w = ibis.window(group_by=col)
        count_cols.append(cur_table[col].count().over(w).name(col_count))
        count_cols.append(ibis.case().when(cur_table[col].count().over(w).name(col_count) > 1, cur_table[col].cast("float32")).else_(ibis.null()).end().name('var_%d_gt' % i))
        cast_cols.append(cur_table[col].cast("float32").name(col))

    for i in range(200):
        col = 'var_%d' % i
        orig_cols.append(col)

    cur_table = cur_table.mutate(count_cols)
    cur_table = cur_table.drop(orig_cols)
    cur_table = cur_table.mutate(cast_cols)


    frame = cur_table.execute()
    t_groupby_merge_where = time.time() - t0
    
    print("t_groupby_merge_where", t_groupby_merge_where)


    t0 = time.time()

    training_part, validation_part = frame[:-10000],frame[-10000:] 

    t_split_time = time.time() - t0
    print("t_split_time", t_split_time)

    return t_groupby_merge_where

def validation():
    validation_precision = 0

    if args.val and not queries_validation_flag:
        print("Validating queries results ...")
        
        queries_validation_flag = True
               
        float_cols = ['var_%s'%i for i in range(200)] + ['var_%s_gt1'%i for i in range(200)]
        training_part_val = training_part.copy()
        training_part_val = training_part_val.sort_values(by=['rowid'])
        validation_part_val = validation_part.copy()
        validation_part_val = validation_part_val.sort_values(by=['rowid'])
        cast_dict = {'var_%s'%i: 'float64' for i in range(200)}
        cast_dict.update({'var_%s_gt1'%i: 'float64' for i in range(200)})
        training_part_val = training_part_val.astype(dtype=cast_dict, copy=False)
        validation_part_val = validation_part_val.astype(dtype=cast_dict, copy=False)
        training_part_val = training_part_val.drop(['rowid'],axis=1)
        validation_part_val = validation_part_val.drop(['rowid'],axis=1)

        validation_result1 = compare_tables(train, training_part_val)
        validation_result2 = compare_tables(valid, validation_part_val)
        queries_validation_results = validation_result1 and validation_result2
        if queries_validation_results:
            print("Queries results are validated!")


def conversion():
    t0 = time.time()
    #global training_part
    #global validation_part
    global train
    global valid
    global training_dmat_part
    global testing_dmat_part
    global y_valid
    
    train = training_part
    valid = validation_part

    x_train = train.drop(['target'],axis=1)
    y_train = train['target']
    x_valid = valid.drop(['target'],axis=1)
    y_valid = valid['target']

    training_dmat_part = xgboost.DMatrix(data=x_train, label=y_train)
    testing_dmat_part = xgboost.DMatrix(data=x_valid, label=y_valid)

    t_conv_to_dmat = time.time() - t0

    return t_conv_to_dmat


def train():
    t0 = time.time()
    global training_dmat_part
    global testing_dmat_part
    global model

    watchlist = [(training_dmat_part, 'eval'), (testing_dmat_part, 'train')]
    xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'max_depth': 1,
            'nthread':56,
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            'eval_metric':'auc',
    }

    model = xgboost.train(xgb_params, dtrain=training_dmat_part,
                num_boost_round=10000, evals=watchlist,
                early_stopping_rounds=30, maximize=True,
                verbose_eval=1000)

    t_train = time.time() - t0

    return t_train


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()

def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ( (y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)

def predict():
    t0 = time.time()
    global testing_dmat_part
    global model
    global y_valid

    yp = model.predict(testing_dmat_part)

    t_inference = time.time() - t0

    score_mse = mse(y_valid, yp)
    score_cod = cod(y_valid, yp)
    print('Scores: ')
    print('  mse = ', score_mse)
    print('  cod = ', score_cod)

    return t_inference


#queries_list = [q1_ibis, q1_FSI, q1_COPY, q2_q3, q4, q5, q6, q7]
queries_list = [etl, conversion, train, predict]
queries_description = {}
queries_description[1] = 'Santander data file import by Ibis query'
#queries_description[2] = 'Santander data file import by SQL COPY stattemrnt query (table with DOUBLE type elements)'
#queries_description[3] = 'Santander data file import by FSI query (table with DOUBLE type elements)'
queries_description[2] = 'Rows group_by+count/merge/filtration query (part of ETL)'
queries_description[3] = 'Rows split query (part of ETL)'
queries_description[4] = 'Conversion to DMatrix'
queries_description[5] = 'ML training'
queries_description[6] = 'ML inference'

omnisci_executable = "../omnisci/build/bin/omnisci_server"
datafile_directory = "/localdisk/work/train.csv"
train_table_name = "train_table"
train_pd_table_name = "train_pd_table"
tmp_table_name = 'tmp_table'
omnisci_server = None
queries_validation_results = False
queries_validation_flag = False


parser = argparse.ArgumentParser(description='Run Santander benchmark using Ibis.')

parser.add_argument('-e', default=omnisci_executable, help='Path to executable "omnisci_server".')
parser.add_argument('-r', default="report_santander_ibis.csv", help="Report file name.")
parser.add_argument('-dp', default=datafile_directory, help="Datafile that should be loaded.")
parser.add_argument('-i', default=1, type=int,
                    help="Number of iterations to run every query. Best result is selected.")
parser.add_argument('-dnd', action='store_true', help="Do not delete old table.")
parser.add_argument('-dni', action='store_true',
                    help="Do not create new table and import any data from CSV files.")
parser.add_argument("-port", default=62074, type=int,
                    help="TCP port that omnisql client should use to connect to server.")
parser.add_argument("-u", default="admin",
                    help="User name to use on omniscidb server.")
parser.add_argument("-p", default="HyperInteractive",
                    help="User password to use on omniscidb server.")
parser.add_argument("-n", default="agent_test_ibis",
                    help="Database name to use on omniscidb server.")
parser.add_argument('-q3_full', action='store_true', help="Execute q3 query correctly (script execution time will be increased).")
parser.add_argument('-val', action='store_true', help="validate queries results (by comparison with Pandas queries results).")

parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server.")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server.")
parser.add_argument("-db-user", default="",
                    help="Username to use to connect to MySQL database. "
                         "If user name is specified, script attempts to store results in "
                         "MySQL database using other -db-* parameters.")
parser.add_argument("-db-pass", default="omniscidb",
                    help="Password to use to connect to MySQL database.")
parser.add_argument("-db-name", default="omniscidb",
                    help="MySQL database to use to store benchmark results.")
parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

parser.add_argument("-commit_omnisci", dest="commit_omnisci",
                    default="1234567890123456789012345678901234567890",
                    help="Omnisci commit hash to use for tests.")
parser.add_argument("-commit_ibis", dest="commit_ibis",
                    default="1234567890123456789012345678901234567890",
                    help="Ibis commit hash to use for tests.")

try:
    args = parser.parse_args()

    if args.i < 1:
        print("Bad number of iterations specified", args.i)

    database_name = args.n
    omnisci_server = OmnisciServer(omnisci_executable=args.e, omnisci_port=args.port,
                                   database_name=database_name, user=args.u,
                                   password=args.p)

    datafile_columns_names = ["ID_code", "target"] + ["var_" + str(index) for index in range(200)]
    datafile_columns_types = ["string", "int64"] + ["decimal(8, 4)" for _ in range(200)]

    # SQL queries preparation
    connect_to_db_sql_query_template = "\c {0} admin HyperInteractive"
    create_table_sql_query_template = '''
    CREATE TABLE {0} ({1});
    '''
    import_csv_by_COPY_sql_query_template = '''
    COPY {0} FROM '{1}' WITH (header='{2}');
    '''
    import_csv_by_FSI_sql_query_template = '''
    CREATE TEMPORARY TABLE {0} ({1}) WITH (storage_type='CSV:{2}');
    '''
    drop_table_sql_query_template = '''
    DROP TABLE IF EXISTS {0};
    '''

    table_cols_list = ["ID_code TEXT ENCODING NONE, \n", "target SMALLINT, \n"] + [
        "var_%s DOUBLE, \n" % i for i in range(199)] + ["var_199 DOUBLE"]
    table_cols_str = "".join(table_cols_list)

    data_file_name, data_file_ext = os.path.splitext(args.dp)
    csv_data_file = args.dp
    if data_file_ext != '.csv':
        csv_data_file = data_file_name
        if not os.path.exists(data_file_name):
           execute_process(cmdline=['tar', '-xvf', args.dp, '--strip 1'], cwd=pathlib.Path(args.e).parent)

    connect_to_db_sql_query = connect_to_db_sql_query_template.format(database_name)
    create_table_sql_query = create_table_sql_query_template.format(tmp_table_name, table_cols_str)
    import_csv_by_COPY_sql_query = import_csv_by_COPY_sql_query_template.format(tmp_table_name, str(args.dp), 'true')
    drop_table_sql_query = drop_table_sql_query_template.format(train_table_name)

    import_csv_by_FSI_sql_query = import_csv_by_FSI_sql_query_template.format(tmp_table_name, table_cols_str, str(csv_data_file))
    drop_tmp_table_sql_query = drop_table_sql_query_template.format(tmp_table_name)


    omnisci_server.launch()
    omnisci_server_worker = OmnisciServerWorker(omnisci_server)

    time.sleep(2)
    conn_ipc = omnisci_server_worker.ipc_connect_to_server()
    conn = omnisci_server_worker.connect_to_server()


    db_reporter = None
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
            'QueryValidation': 'VARCHAR(500) NOT NULL',
            'IbisCommitHash': 'VARCHAR(500) NOT NULL'
        }, {
            'ScriptName': 'santander_ibis.py',
            'CommitHash': args.commit_omnisci
        })

    # Delete old table
    if not args.dnd:
        print("Deleting", database_name, "old database")
        try:
            conn.drop_database(database_name, force=True)
            time.sleep(2)
            conn = omnisci_server_worker.connect_to_server()
        except Exception as err:
            print("Failed to delete", database_name, "old database: ", err)

    print("Creating new database", database_name)
    try:
        conn.create_database(database_name)  # Ibis list_databases method is not supported yet
    except Exception as err:
        print("Database creation is skipped, because of error:", err)

    args.dp = args.dp.replace("'", "")
    if not args.dni:
        # Datafiles import
        t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
            table_name=train_table_name, data_files_names=args.dp, files_limit=1,
            columns_names=datafile_columns_names, columns_types=datafile_columns_types,
            cast_dict=None, header=0)
        print("Pandas import time:", t_import_pandas)
        print("Ibis import time:", t_import_ibis)

    try:
        db = conn_ipc.database(database_name)
    except Exception as err:
        print("Failed to connect to database:", err)

    try:
        tables_names = db.list_tables()
        print("Database tables:", tables_names)
    except Exception as err:
        print("Failed to read database tables:", err)

    try:
        df = db.table(train_table_name)
    except Exception as err:
        print("Failed to access", train_table_name, "table:", err)

#    pandas part - should be outlined in separate routine
    #t0 = time.time()
    #train_pd = pd.read_csv(str(args.dp))
    #t_pandas_import = time.time() - t0
    
    #print("t_import_pandas", t_pandas_import)

    #t0 = time.time()
    #for i in range(200):
    #    col = 'var_%d' % i
    #    var_count = train_pd.groupby(col).agg({col: 'count'})
    #    var_count.columns = ['%s_count' % col]
    #    var_count = var_count.reset_index()
    #    train_pd = train_pd.merge(var_count, on=col, how='left')

    #for i in range(200):
    #    col = 'var_%d' % i
    #    mask = train_pd['%s_count' % col] > 1
    #    train_pd.loc[mask, '%s_gt1' % col] = train_pd.loc[mask, col]
        
    #tmp1, tmp2 = train_pd[:-10000],train_pd[-10000:]        
    #t_pandas_etl = time.time() - t0
    #print("t_pandas_etl", t_pandas_etl)

    try:

        with open(args.r, "w") as report:
            t_begin = time.time()
            for query_number in range(0, 4):
                exec_times = [None] * 4
                best_exec_time = float("inf")
                worst_exec_time = 0.0
                first_exec_time = float("inf")
                times_sum = 0.0
                for iteration in range(0, args.i):
                    print("Running query number:", query_number + 1, "Iteration number:", iteration)
                    time_tmp = int(round(queries_list[query_number]() * 1000))
                    exec_times[iteration - 1] = time_tmp
                    if iteration == 1:
                        first_exec_time = exec_times[iteration - 1]
                    if best_exec_time > exec_times[iteration - 1]:
                        best_exec_time = exec_times[iteration - 1]
                    if iteration != 1 and worst_exec_time < exec_times[iteration - 1]:
                        worst_exec_time = exec_times[iteration - 1]
                    if iteration != 1:
                        times_sum += exec_times[iteration - 1]
                #average_exec_time = times_sum / (args.i - 1)
                total_exec_time = int(round(time.time() - t_begin))
                print("Query", query_number + 1, "Exec time (ms):", best_exec_time,
                      "Total time (s):", total_exec_time)
                print("QueryName: ", queries_description[query_number + 1], ",",
                      "IbisCommitHash", args.commit_ibis, ",",
                      "FirstExecTimeMS: ", first_exec_time, ",",
                      "WorstExecTimeMS: ", worst_exec_time, ",",
                      "BestExecTimeMS: ", best_exec_time, ",",
                 #     "AverageExecTimeMS: ", average_exec_time, ",",
                      "TotalTimeMS: ", total_exec_time, ",",
                      "", '\n', file=report, sep='', end='', flush=True)
                if db_reporter is not None:
                    db_reporter.submit({
                        'QueryName': queries_description[query_number + 1],
                        'IbisCommitHash': args.commit_ibis,
                        'FirstExecTimeMS': first_exec_time,
                        'WorstExecTimeMS': worst_exec_time,
                        'BestExecTimeMS': best_exec_time,
                        'AverageExecTimeMS': str(queries_validation_results['q%s' % (query_number + 1)]),
                        'TotalTimeMS': total_exec_time
                    })
    except IOError as err:
        print("Failed writing report file", args.r, err)
except Exception as exc:
    print("Failed: ", exc)
finally:
    if omnisci_server:
        omnisci_server.terminate()
