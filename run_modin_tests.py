from dataclasses import dataclass
import os
import sys
from typing import Iterable

from utils_base_env import execute_process, prepare_parser


# This can be written as just a function, but we keep the dataclass to add validation and arg parsing in the future.
@dataclass
class DbConfig:
    """Class encapsulates DB configuration and connection."""

    driver: str
    server: str
    port: int
    user: str
    password: str
    name: str

    def create_engine(self):
        from sqlalchemy import create_engine

        url = f"{self.driver}://{self.user}:{self.password}@{self.server}:{self.port}/{self.name}"
        return create_engine(url, future=True)


def parse_tasks(task_string: str, possible_tasks: Iterable[str]):
    required_tasks = task_string.split(",")
    possible_tasks = set(possible_tasks)

    if len(set(required_tasks) - possible_tasks) > 0:
        raise ValueError(
            f"Discovered unrecognized task type. Received {required_tasks}, but only"
            f"{possible_tasks} are supported"
        )

    tasks = [t for t in required_tasks if t in possible_tasks]
    if len(tasks) == 0:
        raise ValueError(
            f"Only {possible_tasks} are supported, received {required_tasks} cannot find any possible task"
        )

    return tasks


def rerun_with_env(args):
    """Activate the environment from the parameters and run the same script again without `--env_name -en` parameter"""
    from environment import CondaEnvironment

    print("PREPARING ENVIRONMENT")
    conda_env = CondaEnvironment(args.env_name)
    conda_env.create(
        python_version=args.python_version,
        existence_check=args.env_check,
        requirements_file=args.ci_requirements,
        channel="conda-forge",
    )
    main_cmd = sys.argv.copy()
    try:
        env_name_idx = main_cmd.index("--env_name")
    except ValueError:
        env_name_idx = main_cmd.index("-en")
    # drop env name: option and value
    drop_env_name = env_name_idx + 2
    main_cmd = ["python3"] + main_cmd[:env_name_idx] + main_cmd[drop_env_name:]
    try:
        data_file_idx = main_cmd.index("-data_file") + 1
        # for some workloads, in the filename, we use "{", "}" characters that the shell
        # itself can expands, for which our interface is not designed;
        # "'" symbols disable expanding arguments by shell
        main_cmd[data_file_idx] = f"'{main_cmd[data_file_idx]}'"
    except ValueError:
        pass

    print(" ".join(main_cmd))
    try:
        # Rerun the command after activating the environment
        conda_env.run(main_cmd)
    finally:
        if args and args.save_env is False:
            conda_env.remove()


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


def run_benchmark_task(args):
    from utils import run_benchmarks

    if not args.data_file:
        raise ValueError(
            "Parameter --data_file was received empty, but it is required for benchmarks"
        )

    db_config = DbConfig(
        driver=args.db_driver,
        server=args.db_server,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        name=args.db_name,
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
        db_config,
        args.commit_hdk,
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
        rerun_with_env(args)
    else:
        # just to ensure that we in right environment
        execute_process(["conda", "env", "list"], print_output=True)

        if "build" in tasks:
            run_build_task(args)

        if "benchmark" in tasks:
            run_benchmark_task(args)


if __name__ == "__main__":
    main()
