import subprocess


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