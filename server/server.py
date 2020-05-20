import os
import pathlib
import signal
import sys
import threading
import time

from utils_base_env import execute_process


class OmnisciServer:
    "Manage interactions with OmniSciDB server (launch/termination, connection establishing, etc.)"

    server_process = None
    _header_santander_train = False

    def __init__(
        self,
        omnisci_executable,
        omnisci_port,
        database_name,
        http_port,
        calcite_port,
        omnisci_cwd=None,
        user="admin",
        password="HyperInteractive",
        max_session_duration=86400,
        idle_session_duration=120,
        debug_timer=False,
        columnar_output=True,
        lazy_fetch=None,
        multifrag_rs=None,
        omnisci_run_kwargs={},
    ):
        if not os.path.isdir(omnisci_executable) and not os.access(omnisci_executable, os.X_OK):
            raise ValueError("Invalid omnisci executable given: " + omnisci_executable)
        self.omnisci_executable = omnisci_executable
        self.server_port = omnisci_port
        self.user = user
        self.password = password
        self.database_name = database_name
        self._http_port = http_port
        self._calcite_port = calcite_port
        self._max_session_duration = max_session_duration
        self._idle_session_duration = idle_session_duration

        if omnisci_cwd is not None:
            self._server_cwd = omnisci_cwd
        else:
            self._server_cwd = pathlib.Path(self.omnisci_executable).parent.parent

        self._data_dir = os.path.join(self._server_cwd, "data")
        if not os.path.isdir(self._data_dir):
            print("CREATING DATA DIR", self._data_dir)
            os.makedirs(self._data_dir)
        if not os.path.isdir(os.path.join(self._data_dir, "mapd_data")):
            print("INITIALIZING DATA DIR", self._data_dir)
            self._initdb_executable = os.path.join(
                pathlib.Path(self.omnisci_executable).parent, "initdb"
            )
            execute_process([self._initdb_executable, "-f", "--data", self._data_dir])

        self.omnisci_sql_executable = os.path.join(
            pathlib.Path(self.omnisci_executable).parent, "omnisql"
        )
        self._server_start_cmdline = [
            self.omnisci_executable,
            "data",
            "--port",
            str(omnisci_port),
            "--http-port",
            str(self._http_port),
            "--calcite-port",
            str(self._calcite_port),
            "--config",
            "omnisci.conf",
            "--enable-watchdog=false",
            "--allow-cpu-retry",
            "--max-session-duration",
            str(self._max_session_duration),
            "--idle-session-duration",
            str(self._idle_session_duration),
        ]

        if debug_timer is not None:
            self._server_start_cmdline.append(
                "--enable-debug-timer=%s" % ("true" if debug_timer else "false")
            )

        if columnar_output is not None:
            self._server_start_cmdline.append(
                "--enable-columnar-output=%s" % ("true" if columnar_output else "false")
            )

        if lazy_fetch is not None:
            self._server_start_cmdline.append(
                "--enable-lazy-fetch=%s" % ("true" if lazy_fetch else "false")
            )

        if multifrag_rs is not None:
            self._server_start_cmdline.append(
                "--enable-multifrag-rs=%s" % ("true" if multifrag_rs else "false")
            )

        for key, value in omnisci_run_kwargs.items():
            self._server_start_cmdline.append(f"--{key}={value}")

    def launch(self):
        "Launch OmniSciDB server"

        print("Launching server ...")
        self.server_process, _ = execute_process(
            self._server_start_cmdline, cwd=self._server_cwd, daemon=True
        )
        print("Server is launched")
        try:
            pt = threading.Thread(
                target=self._print_omnisci_output, args=(self.server_process.stdout,), daemon=True,
            )
            pt.start()

            # Allow server to start up. It has to open TCP port and start
            # listening, otherwise the following benchmarks fail.
            time.sleep(5)
        except Exception as err:
            print("Failed", err)
            sys.exit(1)

    def _print_omnisci_output(self, stdout):
        for line in iter(stdout.readline, b""):
            print("OMNISCI>>", line.decode().strip())

    def terminate(self):
        "Terminate OmniSci server"

        print("Terminating server ...")

        try:
            if self.server_process:
                self.server_process.send_signal(signal.SIGINT)
                time.sleep(2)
                self.server_process.kill()
                time.sleep(1)
                self.server_process = None
        except Exception as err:
            print("Failed to terminate server, error occured:", err)
            sys.exit(1)

        print("Server is terminated")
