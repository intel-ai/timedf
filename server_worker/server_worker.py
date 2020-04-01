import gzip
import os
import subprocess
import sys
import time

import ibis
import pandas as pd
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import convert_type_ibis2pandas


class OmnisciServerWorker:
    _imported_pd_df = {}

    def __init__(self, omnisci_server):
        self.omnisci_server = omnisci_server
        self._command_2_import_CSV = "COPY %s FROM '%s' WITH (header='%s');"
        self._conn = None

    def _get_omnisci_cmd_line(
        self, database_name=None, user=None, password=None, server_port=None
    ):
        if not database_name:
            database_name = self.omnisci_server.database_name
        if not user:
            user = self.omnisci_server.user
        if not password:
            password = self.omnisci_server.password
        if not server_port:
            server_port = str(self.omnisci_server.server_port)
        return (
            [self.omnisci_server.omnisci_sql_executable]
            + [database_name, "-u", user, "-p", password]
            + ["--port", server_port]
        )

    def _read_csv_datafile(
        self,
        file_name,
        columns_names,
        columns_types=None,
        header=None,
        compression_type="gzip",
        nrows=None,
        skiprows=None,
    ):
        "Read csv by Pandas. Function returns Pandas DataFrame,\
        which can be used by ibis load_data function"

        print("Reading datafile", file_name)
        types = None
        if columns_types:
            types = {
                columns_names[i]: columns_types[i] for i in range(len(columns_names))
            }
        if compression_type == "gzip":
            with gzip.open(file_name) as f:
                return pd.read_csv(
                    f, names=columns_names, dtype=types, nrows=nrows, header=header
                )

        return pd.read_csv(
            file_name,
            compression=compression_type,
            names=columns_names,
            dtype=types,
            nrows=nrows,
            header=header,
            skiprows=skiprows,
        )

    def import_data_by_pandas(
        self,
        data_files_names,
        files_limit,
        columns_names,
        nrows=None,
        compression_type="gzip",
    ):
        "Import CSV files using Pandas read_csv to the Pandas.DataFrame"

        if files_limit == 1:
            return self._read_csv_datafile(
                file_name=data_files_names[0],
                columns_names=columns_names,
                header=None,
                compression_type=compression_type,
                nrows=nrows,
            )
        else:
            df_from_each_file = (
                self._read_csv_datafile(
                    file_name=f,
                    columns_names=columns_names,
                    header=None,
                    compression_type=compression_type,
                    nrows=nrows,
                )
                for f in data_files_names[:files_limit]
            )
            return pd.concat(df_from_each_file, ignore_index=True)

    def connect_to_server(self, database=None, ipc=None):
        "Connect to Omnisci server using Ibis framework"

        if self._conn:
            self._conn.close()
        self._conn = ibis.omniscidb.connect(
            host="localhost",
            port=self.omnisci_server.server_port,
            user=self.omnisci_server.user,
            database=database,
            password=self.omnisci_server.password,
            ipc=ipc,
        )
        if database:
            self.omnisci_server.database_name = database

    def get_conn(self):
        return self._conn

    def database(self, name):
        return self._conn.database(name)

    def create_table(self, *arg, **kwargs):
        "Wrapper for OmniSciDBClient.create_table"
        self._conn.create_table(*arg, **kwargs)

    def terminate(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def import_data(
        self,
        table_name,
        data_files_names,
        files_limit,
        columns_names,
        columns_types,
        header=False,
    ):
        "Import CSV files to the OmniSciDB using COPY SQL statement"

        if header:
            header_value = "true"
        elif not header:
            header_value = "false"
        else:
            print("Wrong value of header argument!")
            sys.exit(2)

        schema_table = ibis.Schema(names=columns_names, types=columns_types)

        if not self._conn.exists_table(
            name=table_name, database=self.omnisci_server.database_name
        ):
            try:
                self._conn.create_table(
                    table_name=table_name,
                    schema=schema_table,
                    database=self.omnisci_server.database_name,
                )
            except Exception as err:
                print("Failed to create table:", err)

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % (table_name, f, header_value)

            omnisci_cmd_line = self._get_omnisci_cmd_line()
            try:
                import_process = subprocess.Popen(
                    omnisci_cmd_line,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                )
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print(f"Failed to start '{omnisci_cmd_line}'", err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)

    def import_data_by_ibis(
        self,
        table_name,
        data_files_names,
        files_limit,
        columns_names,
        columns_types,
        cast_dict=None,
        header=None,
        nrows=None,
        compression_type="gzip",
        skiprows=None,
        validation=None,
    ):
        "Import CSV files using Ibis load_data to the OmniSciDB from the Pandas.DataFrame"

        if columns_types:
            columns_types_pd = convert_type_ibis2pandas(columns_types)
        t0 = time.time()
        if files_limit > 1:
            pandas_df_from_each_file = (
                self._read_csv_datafile(
                    file_name,
                    columns_names=columns_names,
                    columns_types=columns_types_pd,
                    header=header,
                    nrows=nrows,
                    compression_type=compression_type,
                )
                for file_name in data_files_names[:files_limit]
            )
            self._imported_pd_df[table_name] = pd.concat(
                pandas_df_from_each_file, ignore_index=True
            )
        else:
            self._imported_pd_df[table_name] = self._read_csv_datafile(
                data_files_names,
                columns_names=columns_names,
                columns_types=columns_types_pd,
                header=header,
                nrows=nrows,
                compression_type=compression_type,
                skiprows=skiprows,
            )
        t_import_pandas = time.time() - t0

        if validation:
            df = self._imported_pd_df[table_name]
            df["id"] = [x + 1 for x in range(df[df.columns[0]].count())]
            columns_names = columns_names + ["id"]
            columns_types = columns_types + ["int32"]
            self._imported_pd_df[table_name] = df

        if cast_dict is not None:
            pandas_concatenated_df_casted = self._imported_pd_df[table_name].astype(
                dtype=cast_dict, copy=True
            )
            t0 = time.time()
            self.import_data_from_pd_df(
                table_name=table_name,
                pd_obj=pandas_concatenated_df_casted,
                columns_names=columns_names,
                columns_types=columns_types,
            )
            t_import_ibis = time.time() - t0
        else:
            t0 = time.time()
            self.import_data_from_pd_df(
                table_name=table_name,
                pd_obj=self._imported_pd_df[table_name],
                columns_names=columns_names,
                columns_types=columns_types,
            )
            t_import_ibis = time.time() - t0

        return t_import_pandas, t_import_ibis

    def drop_table(self, table_name):
        "Drop table by table_name using Ibis framework"

        print("Deleting ", table_name, " table")
        if self._conn.exists_table(
            name=table_name, database=self.omnisci_server.database_name
        ):
            db = self._conn.database(self.omnisci_server.database_name)
            df = db.table(table_name)
            df.drop()
            if table_name in self._imported_pd_df:
                del self._imported_pd_df[table_name]
        else:
            print("Table ", table_name, " doesn't exist!")

    def drop_database(self, database_name, force=True):
        "Drop database by database_name using Ibis framework"

        if self._conn is None:
            self.connect_to_server()

        print("Deleting ", database_name, " database")
        try:
            self._conn.drop_database(database_name, force=force)
        except Exception as err:
            print("Failed to delete ", database_name, "database: ", err)

        self.connect_to_server()

    def create_database(self, database_name, delete_if_exists=True):
        "Create database by database_name using Ibis framework"

        if self._conn is None:
            self.connect_to_server()

        if delete_if_exists:
            self.drop_database(database_name, force=True)
        print("Creating ", database_name, " database")
        try:
            self._conn.create_database(database_name)
            self._conn.set_database(database_name)
        except Exception as err:
            print("Failed to create ", database_name, " database: ", err)

    def get_pd_df(self, table_name):
        "Get already imported Pandas DataFrame"

        if self._conn.exists_table(
            name=table_name, database=self.omnisci_server.database_name
        ):
            return self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(4)

    def execute_sql_query(self, query):
        "Execute SQL query directly in the OmniSciDB"

        omnisci_cmd_line = self._get_omnisci_cmd_line()
        try:
            connection_process = subprocess.Popen(
                omnisci_cmd_line,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
            )
            output = connection_process.communicate(query.encode())
            print(output)
        except OSError as err:
            print("Failed to start", omnisci_cmd_line, err)

    def import_data_from_pd_df(self, table_name, pd_obj, columns_names, columns_types):
        "Import table data using Ibis load_data to the OmniSciDB from the Pandas.DataFrame"

        schema_table = ibis.Schema(names=columns_names, types=columns_types)

        if not self._conn.exists_table(name=table_name):
            try:
                self._conn.create_table(table_name=table_name, schema=schema_table)
            except Exception as err:
                print("Failed to create table:", err)

        self._conn.load_data(
            table_name=table_name,
            obj=pd_obj,
            database=self.omnisci_server.database_name,
            method="columnar",
        )

        return self._conn.database(self.omnisci_server.database_name).table(table_name)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, trace):
        try:
            self.terminate()
        except Exception as err:
            print('terminate is not successful')
            raise err

    def fsi_read_csv(self, table_name, import_query_cols, filename):
        import_by_FSI_sql_template = """
        CREATE TEMPORARY TABLE {0} ({1}) WITH (storage_type='CSV:{2}');
        """

        import_query_cols_str = "".join(import_query_cols)

        import_by_FSI_sql = import_by_FSI_sql_template.format(
            table_name, import_query_cols_str, filename
        )

        t0 = timer()
        self._conn._execute(import_by_FSI_sql)
        return timer() - t0
