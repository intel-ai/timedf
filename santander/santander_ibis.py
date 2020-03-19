import mysql.connector
import argparse
import time
import sys
import os
import ibis
import xgboost

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
from server_worker import OmnisciServerWorker

def compare_tables(table1, table2):
    
    if table1.equals(table2):
        return True
    else:
        print("\ntables are not equal, table1:")
        print(table1.info())
        print("\ntable2:")
        print(table2.info())
        return False

# Queries definitions
def q1():
    t0 = time.time()
    omnisci_server_worker.import_data_by_ibis(table_name=tmp_table_name,
                                                     data_files_names=args.dp, files_limit=1,
                                                     columns_names=datafile_columns_names,
                                                     columns_types=datafile_columns_types,
                                                     cast_dict=None, header=0)
    t_import = time.time() - t0
    
    if args.val and not queries_validation_flags['q1']:
        print("Validating query 1 (import query) results ...")
        
        queries_validation_flags['q1'] = True
        pd_df = omnisci_server_worker.get_pd_df(tmp_table_name)
        ibis_df = conn.database(database_name).table(tmp_table_name)
        queries_validation_results['q1'] = compare_tables(pd_df, ibis_df.execute())
        if queries_validation_results['q1']:
            print("q1 results are validated!")
            
        
    omnisci_server_worker.drop_table(tmp_table_name)

    return t_import


def q2():
    t_groupby = 0
    if args.val and not queries_validation_flags['q2']:
        print("Validating query 2 (group_by query) results ...")
        compare_results_full = True
        train_pd_group_by = omnisci_server_worker.get_pd_df(train_table_name)
    
    for i in range(200):
        col = 'var_%d' % i
        t0 = time.time()
        metric = df[col].count().name('%s_count' % col)
        #group_by_expr = df.group_by(col).order_by(df[col]).aggregate(metric) # - doesn't give correct result
        group_by_expr = df.sort_by(col).group_by(col).aggregate(metric)
        var_count_ibis = group_by_expr.execute()
        t_groupby += time.time() - t0
        
        if args.val and not queries_validation_flags['q2']:
            
            if i % 10 == 0:
                var_count_pd = train_pd_group_by.groupby(col).agg({col:'count'})
                var_count_pd.columns = ['%s_count'%col]
                var_count_pd = var_count_pd.reset_index()

                compare_results = compare_tables(var_count_pd, var_count_ibis)
                compare_results_full = compare_results_full and compare_results
            if i == 199:
                queries_validation_flags['q2'] = True
                queries_validation_results['q2'] = compare_results_full
                if compare_results_full:
                    print("q2 results are validated!")
                
    return t_groupby


def q3():
    t_where = 0
    col_to_sel = datafile_columns_names + ["var_" + str(index) + "_count" for index in range(200)]
    
    if args.q3_full:
        col_to_sel_list = ["var_%s, \n" % i for i in range(200)] + ["var_%s_count, \n" % i for i in range(199)] + ["var_199_count"]
        col_to_sel_str = "".join(col_to_sel_list)
        drop_tmp_table_query = drop_table_sql_query_template.format(database_name, tmp_table_name)
        create_train_where_ibis_query = create_table_sql_query_template.format(database_name, tmp_table_name,
                                                                                    col_to_sel_str, train_pd_table_name)
        omnisci_server_worker.execute_sql_query(create_train_where_ibis_query)
        train_where_ibis = conn.database(database_name).table(tmp_table_name)
    else:
        train_where_ibis = train_pd_ibis[col_to_sel]
        
    if args.val and not queries_validation_flags['q3']:
        print("Validating query 3 (filter query) results ...")
        compare_results_full = True
        train_pd_merged_val = train_pd_merged.copy()
        
    for i in range(200):
        col = 'var_%d' % i
        t0 = time.time()
        mask_ibis = (train_where_ibis['%s_count' % col] > 1).execute()
        t_where += time.time() - t0
        
        if args.val and not queries_validation_flags['q3']:
            mask_pd = train_pd_merged_val['%s_count'%col]>1
            train_pd_merged_val.loc[mask_pd,'%s_gt1'%col] = train_pd_merged_val.loc[mask_pd,col]
            
            if i % 10 == 0:
                compare_results = compare_tables(mask_pd, mask_ibis)
                compare_results_full = compare_results_full and compare_results
            if i == 199:
                queries_validation_flags['q3'] = True
                queries_validation_results['q3'] = compare_results_full
                if compare_results_full:
                    print("q3 results are validated!")

        if args.q3_full:
            omnisci_server_worker.execute_sql_query(drop_tmp_table_query)
            
            if i == 0:
                col_to_sel_str += ',\n%s_gt1' % col
            else:
                col_to_sel_str += ',\n%s_gt1' % col
            create_train_where_ibis_query = create_table_sql_query_template.format(database_name, tmp_table_name,
                                                                                    col_to_sel_str, train_pd_table_name)
            omnisci_server_worker.execute_sql_query(create_train_where_ibis_query)
            train_where_ibis = conn.database(database_name).table(tmp_table_name)
        else:
            col_to_sel += ['%s_gt1' % col]
            train_where_ibis = train_pd_ibis[col_to_sel]

    if args.q3_full:
        omnisci_server_worker.execute_sql_query(drop_tmp_table_query)
        
    return t_where


def q4():
    t0 = time.time()

    # Split operation syntax: OmniSciDBTable[number of rows to split: the last row index of splitted table (last element is not included)]
    training_part = train_pd_ibis[190000:190000].execute()
    validation_part = train_pd_ibis[10000:200000].execute()
    t_split = time.time() - t0
    
    if args.val and not queries_validation_flags['q4']:
        print("Validating query 4 (rows split query) results ...")
        
        queries_validation_flags['q4'] = True
        train,valid = train_pd[:-10000],train_pd[-10000:]

        validation_result1 = compare_tables(train, training_part)
        validation_result2 = compare_tables(valid, validation_part)
        queries_validation_results['q4'] = validation_result1 and validation_result2
        if queries_validation_results['q4']:
            print("q4 results are validated!")

    return t_split


def q5():
    t0 = time.time()
    global training_dmat_part
    global testing_dmat_part
    global y_valid
    
    train_q5,valid_q5 = train_pd[:-10000],train_pd[-10000:]

    x_train = train_q5.drop(['target','ID_code'],axis=1)
    y_train = train_q5['target']
    x_valid = valid_q5.drop(['target','ID_code'],axis=1)
    y_valid = valid_q5['target']

    training_dmat_part = xgboost.DMatrix(data=x_train, label=y_train)
    testing_dmat_part = xgboost.DMatrix(data=x_valid, label=y_valid)

    t_conv_to_dmat = time.time() - t0

    return t_conv_to_dmat


def q6():
    t0 = time.time()
    global training_dmat_part
    global testing_dmat_part
    global model

    watchlist = [(testing_dmat_part, 'eval'), (training_dmat_part, 'train')]
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

def q7():
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


queries_list = [q1, q2, q3, q4, q5, q6, q7]
queries_description = {}
queries_description[1] = 'Santander data file import query'
queries_description[2] = 'Ibis group_gy and count query'
queries_description[3] = 'Rows filtration query'
queries_description[4] = 'Rows split query'
queries_description[5] = 'Conversion to DMatrix'
queries_description[6] = 'ML training'
queries_description[7] = 'ML inference'

omnisci_executable = "../omnisci/build/bin/omnisci_server"
datafile_directory = "/localdisk/work/train.csv"
train_table_name = "train_table"
train_pd_table_name = "train_pd_table"
tmp_table_name = 'tmp_table'
omnisci_server = None
queries_validation_results = {'q%s' % i: False for i in range(1, 5)}
queries_validation_results.update({'q%s' % i: "validation operation is not supported" for i in range(5, 8)})
queries_validation_flags = {'q%s' % i: False for i in range(1, 8)}
create_table_sql_query_template = '''
\c {0} admin HyperInteractive
CREATE TABLE {1} AS (SELECT {2} FROM {3});
'''
drop_table_sql_query_template = '''
\c {0} admin HyperInteractive
DROP TABLE IF EXISTS {1};
'''

parser = argparse.ArgumentParser(description='Run Santander benchmark using Ibis.')

parser.add_argument('-e', default=omnisci_executable, help='Path to executable "omnisci_server".')
parser.add_argument('-r', default="report_santander_ibis.csv", help="Report file name.")
parser.add_argument('-dp', default=datafile_directory, help="Datafile that should be loaded.")
parser.add_argument('-i', default=5, type=int,
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

    datafile_columns_names = ["ID_code", "target"] + ["var_" + str(index) for index in range(200)]
    datafile_columns_types = ["string", "int64"] + ["float64" for _ in range(200)]

    schema_train = ibis.Schema(
        names=datafile_columns_names,
        types=datafile_columns_types
    )

    database_name = args.n
    omnisci_server = OmnisciServer(omnisci_executable=args.e, omnisci_port=args.port,
                                   database_name=database_name, user=args.u,
                                   password=args.p)
    omnisci_server.launch()
    omnisci_server_worker = OmnisciServerWorker(omnisci_server)

    time.sleep(2)
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

    print("Creating new database")
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
        db = conn.database(database_name)
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

    # Since OmniSciDB doesn't support JOIN operation for tables with non-integer
    # values, tables for filter and split queries were reproduced by Pandas (as it
    # it was done in the similar Pandas benchmark

    train_pd = omnisci_server_worker.get_pd_df(table_name=train_table_name)

    for i in range(200):
        col = 'var_%d' % i
        var_count = train_pd.groupby(col).agg({col: 'count'})
        var_count.columns = ['%s_count' % col]
        var_count = var_count.reset_index()
        train_pd = train_pd.merge(var_count, on=col, how='left')

    train_pd_merged = train_pd.copy()

    for i in range(200):
        col = 'var_%d' % i
        mask = train_pd['%s_count' % col] > 1
        train_pd.loc[mask, '%s_gt1' % col] = train_pd.loc[mask, col]

    datafile_columns_names_train_pd = datafile_columns_names + [
        "var_" + str(index) + "_count" for index in range(200)] + [
        "var_" + str(index) + "_gt1" for index in range(200)]
    datafile_columns_types_train_pd = datafile_columns_types + [
        "int64" for _ in range(200)] + [
        "float64" for _ in range(200)]

    train_pd_ibis = omnisci_server_worker.import_data_from_pd_df(table_name=train_pd_table_name, pd_obj=train_pd,
                                                                 columns_names=datafile_columns_names_train_pd,
                                                                 columns_types=datafile_columns_types_train_pd)


    try:
        with open(args.r, "w") as report:
            t_begin = time.time()
            for query_number in range(0, 7):
                exec_times = [None] * 5
                best_exec_time = float("inf")
                worst_exec_time = 0.0
                first_exec_time = float("inf")
                times_sum = 0.0
                for iteration in range(1, args.i + 1):
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
                average_exec_time = times_sum / (args.i - 1)
                total_exec_time = int(round(time.time() - t_begin))
                print("Query", query_number + 1, "Exec time (ms):", best_exec_time,
                      "Total time (s):", total_exec_time)
                print("QueryName: ", queries_description[query_number + 1], ",",
                      "IbisCommitHash", args.commit_ibis, ",",
                      "FirstExecTimeMS: ", first_exec_time, ",",
                      "WorstExecTimeMS: ", worst_exec_time, ",",
                      "BestExecTimeMS: ", best_exec_time, ",",
                      "AverageExecTimeMS: ", average_exec_time, ",",
                      "QueryValidation: ", str(queries_validation_results['q%s' % (query_number + 1)]), ",",
                      "TotalTimeMS: ", total_exec_time, ",",
                      "", '\n', file=report, sep='', end='', flush=True)
                if db_reporter is not None:
                    db_reporter.submit({
                        'QueryName': queries_description[query_number + 1],
                        'FirstExecTimeMS': first_exec_time,
                        'WorstExecTimeMS': worst_exec_time,
                        'BestExecTimeMS': best_exec_time,
                        'AverageExecTimeMS': average_exec_time,
                        'TotalTimeMS': total_exec_time * 1000,
                        'IbisCommitHash': args.commit_ibis
                    })
    except IOError as err:
        print("Failed writing report file", args.r, err)
except Exception as exc:
    print("Failed: ", exc)
finally:
    if omnisci_server:
        omnisci_server.terminate()
