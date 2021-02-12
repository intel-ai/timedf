import argparse
import re
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
            if re.findall(r"\d fail", output) or re.findall(r"[e,E]rror", output):
                process.returncode = 1
            elif print_output:
                print(output)
        if process.returncode != 0 and process.returncode is not None:
            raise Exception(f"Command returned {process.returncode}. \n{output}")
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
