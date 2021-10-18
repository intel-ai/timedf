import gzip
import subprocess
from timeit import default_timer as timer

import pandas as pd

class OmnisciServerWorker:
    _imported_pd_df = {}

    def __init__(self, omnisci_server):
        self.omnisci_server = omnisci_server
        self._command_2_import_CSV = "COPY %s FROM '%s' WITH (header='%s');"
        self._conn = None
        self._conn_creation_time = 0.0

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
        "Read csv by Pandas. Function returns Pandas DataFrame"

        print("Reading datafile", file_name)
        types = None
        if columns_types:
            types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
        if compression_type == "gzip":
            with gzip.open(file_name) as f:
                return pd.read_csv(f, names=columns_names, dtype=types, nrows=nrows, header=header)

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
        self, data_files_names, files_limit, columns_names, nrows=None, compression_type="gzip"
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

    def get_pd_df(self, table_name):
        "Get already imported Pandas DataFrame"

        if self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            return self._imported_pd_df[table_name]
        else:
            raise ValueError("Table", table_name, "doesn't exist!")

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


    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, trace):
        try:
            self.terminate()
        except Exception as err:
            print("terminate is not successful")
            raise err

    def get_conn_creation_time(self):
        return self._conn_creation_time
