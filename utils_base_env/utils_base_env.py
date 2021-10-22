import argparse
import socket
import subprocess

returned_port_numbers = []


def str_arg_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Cannot recognize boolean value.")


def execute_process(cmdline, cwd=None, shell=False, daemon=False, print_output=True):
    "Execute cmdline in user-defined directory by creating separated process"
    try:
        print("CMD: ", " ".join(cmdline))
        output = ""
        process = subprocess.Popen(
            cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell
        )
        if not daemon:
            output = process.communicate()[0].strip().decode()
        # No `None` value indicates that the process has terminated
        if process.returncode is not None:
            if process.returncode != 0:
                raise Exception(f"{output}\n\nCommand returned {process.returncode}.")
            if print_output:
                print(output)
        return process, output
    except OSError as err:
        print("Failed to start", cmdline, err)


def check_port_availability(port_num):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port_num))
    except Exception:
        return False
    finally:
        sock.close()
    return True


def find_free_port():
    min_port_num = 49152
    max_port_num = 65535
    if len(returned_port_numbers) == 0:
        port_num = min_port_num
    else:
        port_num = returned_port_numbers[-1] + 1
    while port_num < max_port_num:
        if check_port_availability(port_num) and port_num not in returned_port_numbers:
            returned_port_numbers.append(port_num)
            return port_num
        port_num += 1
    raise Exception("Can't find available ports")


class KeyValueListParser(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            kwargs[k] = v
        setattr(namespace, self.dest, kwargs)


def add_mysql_arguments(parser, etl_ml_tables=False):
    parser.add_argument(
        "-db_server", dest="db_server", default="localhost", help="Host name of MySQL server."
    )
    parser.add_argument(
        "-db_port", dest="db_port", default=3306, type=int, help="Port number of MySQL server."
    )
    parser.add_argument(
        "-db_user",
        dest="db_user",
        help="Username to use to connect to MySQL database. "
        "If user name is specified, script attempts to store results in MySQL "
        "database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db_pass",
        dest="db_pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database.",
    )
    parser.add_argument(
        "-db_name",
        dest="db_name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results.",
    )
    if etl_ml_tables:
        parser.add_argument(
            "-db_table_etl",
            dest="db_table_etl",
            help="Table to use to store ETL results for this benchmark.",
        )
        parser.add_argument(
            "-db_table_ml",
            dest="db_table_ml",
            help="Table to use to store ML results for this benchmark.",
        )
    else:
        parser.add_argument(
            "-db-table", dest="db_table", help="Table to use to store results for this benchmark."
        )
