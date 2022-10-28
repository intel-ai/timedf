import os
import sys
from typing import Iterable

from utils_base_env import execute_process, prepare_parser


def parse_tasks(task_string: str, possible_tasks: Iterable[str]):
    required_tasks = set(task_string.split(","))
    possible_tasks = set(possible_tasks)
    
    if len(required_tasks - possible_tasks) > 0:
        raise ValueError(f"Discovered unrecognized task type. Received {required_tasks}, but only"
                         f"{possible_tasks} are supported")

    tasks = possible_tasks.intersection(required_tasks)
    if len(tasks) == 0:
        raise ValueError(f"Only {possible_tasks} are supported, received {required_tasks} cannot find any possible task")

    return tasks

def run_build_task(args):
    if args.modin_path:
        if args.modin_pkgs_dir:
            os.environ["PYTHONPATH"] = (
                os.getenv("PYTHONPATH") + os.pathsep + args.modin_pkgs_dir
                if os.getenv("PYTHONPATH")
                else args.modin_pkgs_dir
            )

        install_cmdline_modin_pip = ["pip", "install", ".[ray]"]

        print("MODIN INSTALLATION")
        execute_process(install_cmdline_modin_pip, cwd=args.modin_path)

    # trying to install dbe extension if omnisci generated it
    executables_path = os.path.dirname(args.executable)
    dbe_path = os.path.join(os.path.abspath(f"{executables_path}/.."), "Embedded")

    if os.path.exists(dbe_path):
        print("DBE INSTALLATION")
        
        omniscidb_root = os.path.abspath(f"{executables_path}/../../")
        for component in ("DBE", "QE", "thrift", "jar",):
            cmake_cmdline = [
                "cmake",
                "--install",
                "build",
                "--component",
                component,
                "--prefix",
                "$CONDA_PREFIX",
            ]
        
            execute_process(cmake_cmdline, cwd=omniscidb_root)
        execute_process(["python3", "setup.py", "install"], cwd=dbe_path)
    else:
        print("Using Omnisci server")


def run_benchmark_task(args):
    from utils import run_benchmarks

    if not args.data_file:
        raise ValueError(
            "Parameter --data_file was received empty, but it is required for benchmarks"
        )

    run_benchmarks(
        args.bench_name,
        args.data_file,
        args.dfiles_num,
        args.iterations,
        args.validation,
        args.optimizer,
        args.pandas_mode,
        args.ray_tmpdir,
        args.ray_memory,
        args.no_ml,
        args.use_modin_xgb,
        args.gpu_memory,
        args.extended_functionality,
        args.db_server,
        args.db_port,
        args.db_user,
        args.db_pass,
        args.db_name,
        args.db_table_etl,
        args.db_table_ml,
        args.executable,
        args.commit_omnisci,
        args.commit_omniscripts,
        args.commit_modin,
    )


def main(raw_args=None):
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser, possible_tasks, omniscript_path = prepare_parser()
    args = parser.parse_args(raw_args)
    tasks = parse_tasks(args.task, possible_tasks=possible_tasks)

    if args.python_version not in ["3.8"]:
        raise NotImplementedError(
            f"Only 3.8 python version is supported, {args.python_version} is not supported"
        )

    if args.env_name is not None:
        from environment import CondaEnvironment

        print("PREPARING ENVIRONMENT")
        conda_env = CondaEnvironment(args.env_name)
        conda_env.create(
            python_version=args.python_version,
            existence_check=args.env_check,
            requirements_file=args.ci_requirements,
            channel="conda-forge",
        )
        test_cmd = sys.argv.copy()
        try:
            env_name_idx = test_cmd.index("--env_name")
        except ValueError:
            env_name_idx = test_cmd.index("-en")
        # drop env name: option and value
        drop_env_name = env_name_idx + 2
        test_cmd = ["python3"] + test_cmd[:env_name_idx] + test_cmd[drop_env_name:]
        try:
            data_file_idx = test_cmd.index("-data_file") + 1
            # for some workloads, in the filename, we use "{", "}" characters that the shell
            # itself can expands, for which our interface is not designed;
            # "'" symbols disable expanding arguments by shell
            test_cmd[data_file_idx] = f"'{test_cmd[data_file_idx]}'"
        except ValueError:
            pass

        print(" ".join(test_cmd))
        try:
            conda_env.run(test_cmd)
        finally:
            if args and args.save_env is False:
                conda_env.remove()
        # FIXME: why would we stop at this point?
        return

    # just to ensure that we in right environment
    execute_process(["conda", "env", "list"], print_output=True)

    if "build" in tasks:
        run_build_task(args)
    
    if "benchmark" in tasks:
        run_benchmark_task(args)


if __name__ == "__main__":
    main()
