import pandas as pd
import subprocess
import gzip
import time
import os
import ibis
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import convertTypeIbis2Pandas


class OmnisciServerWorker:
    _imported_pd_df = {}

    def __init__(self, omnisci_server):
        self.omnisci_server = omnisci_server
        self._omnisci_cmd_line = [self.omnisci_server.omnisci_sql_executable] +\
                                 [str(self.omnisci_server.database_name),
                                  "-u", self.omnisci_server.user,
                                  "-p", self.omnisci_server.password] + \
                                 ["--port", str(self.omnisci_server.server_port)]
        self._command_2_import_CSV = "COPY %s FROM '%s' WITH (header='%s');"
        self._conn = None

    def _read_csv_datafile(self, file_name, columns_names, columns_types=None, header=None,
                           compression_type='gzip', nrows=None, skiprows=None):
        "Read csv by Pandas. Function returns Pandas DataFrame,\
        which can be used by ibis load_data function"

        print("Reading datafile", file_name)
        types = None
        if columns_types:
            types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
        if compression_type == 'gzip':
            with gzip.open(file_name) as f:
                return pd.read_csv(f, names=columns_names, dtype=types, nrows=nrows, header=header)
        return pd.read_csv(file_name, compression=compression_type, names=columns_names,
                           dtype=types,
                           nrows=nrows, header=header, skiprows=skiprows)

    def import_data_by_pandas(self, data_files_names, files_limit, columns_names, nrows=None,
                              compression_type='gzip'):
        "Import CSV files using Pandas read_csv to the Pandas.DataFrame"

        if files_limit == 1:
            return self._read_csv_datafile(file_name=data_files_names[0],
                                           columns_names=columns_names,
                                           header=None, compression_type=compression_type,
                                           nrows=nrows)
        else:
            df_from_each_file = (self._read_csv_datafile(file_name=f, columns_names=columns_names,
                                                         header=None,
                                                         compression_type=compression_type,
                                                         nrows=nrows)
                                 for f in data_files_names[:files_limit])
            return pd.concat(df_from_each_file, ignore_index=True)

    def connect_to_server(self):
        "Connect to Omnisci server using Ibis framework"

        self._conn = ibis.omniscidb.connect(host="localhost", port=self.omnisci_server.server_port,
                                            user=self.omnisci_server.user,
                                            password=self.omnisci_server.password)
        return self._conn

    def terminate(self):
        self.omnisci_server.terminate()

    def import_data(self, table_name, data_files_names, files_limit, columns_names, columns_types,
                    header=False):
        "Import CSV files to the OmniSciDB using COPY SQL statement"

        if header:
            header_value = 'true'
        elif not header:
            header_value = 'false'
        else:
            print("Wrong value of header argument!")
            sys.exit(2)

        schema_table = ibis.Schema(
            names=columns_names,
            types=columns_types
        )

        if not self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            try:
                self._conn.create_table(table_name=table_name, schema=schema_table,
                                        database=self.omnisci_server.database_name)
            except Exception as err:
                print("Failed to create table:", err)

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % (table_name, f, header_value)

            try:
                import_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print("Failed to start", self._omnisci_cmd_line, err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)

    def import_data_by_ibis(self, table_name, data_files_names, files_limit, columns_names,
                            columns_types, cast_dict=None, header=None, nrows=None,
                            compression_type='gzip', skiprows=None):
        "Import CSV files using Ibis load_data to the OmniSciDB from the Pandas.DataFrame"

        if columns_types:
            columns_types_pd = convertTypeIbis2Pandas(columns_types)
        t0 = time.time()
        if files_limit > 1:
            pandas_df_from_each_file = (
                self._read_csv_datafile(file_name, columns_names=columns_names,
                                        columns_types=columns_types_pd,
                                        header=header, nrows=nrows, compression_type=compression_type)
                for file_name in data_files_names[:files_limit])
            self._imported_pd_df[table_name] = pd.concat(pandas_df_from_each_file,
                                                         ignore_index=True)
        else:
            self._imported_pd_df[table_name] = self._read_csv_datafile(data_files_names,
                                                                       columns_names=columns_names,
                                                                       columns_types=columns_types_pd,
                                                                       header=header, nrows=nrows,
                                                                       compression_type=compression_type,
                                                                       skiprows=skiprows)
        t_import_pandas = time.time() - t0
        if cast_dict is not None:
            pandas_concatenated_df_casted = self._imported_pd_df[table_name].astype(dtype=cast_dict,
                                                                                    copy=True)
            t0 = time.time()
            self.import_data_from_pd_df(table_name=table_name, pd_obj=pandas_concatenated_df_casted,
                                        columns_names=columns_names, columns_types=columns_types)
            t_import_ibis = time.time() - t0
        else:
            t0 = time.time()
            self.import_data_from_pd_df(table_name=table_name,
                                        pd_obj=self._imported_pd_df[table_name],
                                        columns_names=columns_names, columns_types=columns_types)
            t_import_ibis = time.time() - t0

        return t_import_pandas, t_import_ibis

    def drop_table(self, table_name):
        "Drop table by table_name using Ibis framework"

        print("Deleting ", table_name, " table")
        if self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            db = self._conn.database(self.omnisci_server.database_name)
            df = db.table(table_name)
            df.drop()
            if table_name in self._imported_pd_df:
                del self._imported_pd_df[table_name]
        else:
            print("Table ", table_name, " doesn't exist!")

    def drop_database(self, database_name, force=True):
        "Drop database by database_name using Ibis framework"

        print("Deleting ", database_name, " database")
        try:
            self._conn.drop_database(database_name, force=force)
            time.sleep(2)
            self._conn = self.connect_to_server()
        except Exception as err:
            print("Failed to delete ", database_name, "database: ", err)

    def create_database(self, database_name, delete_if_exists=True):
        "Create database by database_name using Ibis framework"

        if delete_if_exists:
            self.drop_database(database_name, force=True)
        print("Creating ", database_name, " database")
        try:
            self._conn.create_database(database_name)
            time.sleep(2)
        except Exception as err:
            print("Failed to create ", database_name, " database: ", err)

    def get_pd_df(self, table_name):
        "Get already imported Pandas DataFrame"

        if self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            return self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(4)

    def execute_sql_query(self, query):
        "Execute SQL query directly in the OmniSciDB"

        try:
            connection_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
            output = connection_process.communicate(query.encode())
            print(output)
        except OSError as err:
            print("Failed to start", self._omnisci_cmd_line, err)

    def import_data_from_pd_df(self, table_name, pd_obj, columns_names, columns_types):
        "Import table data using Ibis load_data to the OmniSciDB from the Pandas.DataFrame"

        schema_table = ibis.Schema(
            names=columns_names,
            types=columns_types
        )

        if not self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            try:
                self._conn.create_table(table_name=table_name, schema=schema_table,
                                        database=self.omnisci_server.database_name)
            except Exception as err:
                print("Failed to create table:", err)

        self._conn.load_data(table_name=table_name, obj=pd_obj,
                             database=self.omnisci_server.database_name, method='columnar')

        return self._conn.database(self.omnisci_server.database_name).table(table_name)
