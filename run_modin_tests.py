import os
import re
import sys

from utils_base_env import execute_process, prepare_parser


def main():
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser, possible_tasks, omniscript_path = prepare_parser()
    args = parser.parse_args()

    required_tasks = args.task.split(",")
    tasks = {}
    for task in possible_tasks:
        tasks[task] = True if task in required_tasks else False

    if True not in list(tasks.values()):
        raise ValueError(
            f"Only {list(tasks.keys())} are supported, {required_tasks} cannot find possible tasks"
        )

    if args.python_version not in ["3.7", "3,6"]:
        raise NotImplementedError(
            f"Only 3.7 and 3.6 python versions are supported, {args.python_version} is not supported"
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
        return

    # just to ensure that we in right environment
    execute_process(["conda", "env", "list"], print_output=True)

    if tasks["build"]:
        install_cmdline = ["python3", "setup.py", "install"]

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
        initdb_path = os.path.join(executables_path, "initdb")
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        initdb_cmdline = [initdb_path, "--data", data_dir]

        if not os.path.isdir(data_dir) and args.manage_dbe_dir:
            print("MANAGING OMNISCI DATA DIR", data_dir)
            os.makedirs(data_dir)
            execute_process(initdb_cmdline)

        if os.path.exists(dbe_path):
            print("DBE INSTALLATION")
            cmake_cmdline = [
                "cmake",
                "--install",
                "build",
                "--component",
                "DBE",
                "--prefix",
                "$CONDA_PREFIX",
            ]
            cmake_qe_cmdline = [
                "cmake",
                "--install",
                "build",
                "--component",
                "QE",
                "--prefix",
                "$CONDA_PREFIX",
            ]
            cmake_thrift_cmdline = [
                "cmake",
                "--install",
                "build",
                "--component",
                "thrift",
                "--prefix",
                "$CONDA_PREFIX",
            ]
            cmake_jar_cmdline = [
                "cmake",
                "--install",
                "build",
                "--component",
                "jar",
                "--prefix",
                "$CONDA_PREFIX",
            ]
            omniscidb_root = os.path.abspath(f"{executables_path}/../../")
            execute_process(cmake_cmdline, cwd=omniscidb_root)
            execute_process(cmake_qe_cmdline, cwd=omniscidb_root)
            execute_process(cmake_thrift_cmdline, cwd=omniscidb_root)
            execute_process(cmake_jar_cmdline, cwd=omniscidb_root)
            execute_process(install_cmdline, cwd=dbe_path)
        else:
            print("Using Omnisci server")

    if tasks["benchmark"]:
        # if not args.bench_name or args.bench_name not in benchmarks:
        #     print(
        #     f"Benchmark {args.bench_name} is not supported, only {benchmarks} are supported")
        # sys.exit(1)

        if not args.data_file:
            raise ValueError(
                "Parameter --data_file was received empty, but it is required for benchmarks"
            )

        benchmark_script_path = os.path.join(omniscript_path, "run_modin_benchmark.py")

        benchmark_cmd = ["python3", benchmark_script_path]

        possible_benchmark_args = [
            "bench_name",
            "data_file",
            "dfiles_num",
            "iterations",
            "validation",
            "optimizer",
            "pandas_mode",
            "ray_tmpdir",
            "ray_memory",
            "no_ml",
            "use_modin_xgb",
            "gpu_memory",
            "db_server",
            "db_port",
            "db_user",
            "db_pass",
            "db_name",
            "db_table_etl",
            "db_table_ml",
            "executable",
            "commit_omnisci",
            "commit_omniscripts",
            "extended_functionality",
            "commit_modin",
        ]
        args_dict = vars(args)
        if not args_dict["data_file"].startswith("'"):
            args_dict["data_file"] = "'{}'".format(args_dict["data_file"])
        for arg_name in list(parser._option_string_actions.keys()):
            try:
                pure_arg = re.sub(r"^--*", "", arg_name)
                if pure_arg in possible_benchmark_args:
                    arg_value = args_dict[pure_arg]
                    # correct filling of arguments with default values
                    if arg_value is not None:
                        if isinstance(arg_value, dict):
                            if arg_value:
                                benchmark_cmd.extend(
                                    [
                                        arg_name,
                                        ",".join(
                                            f"{key}={value}" for key, value in arg_value.items()
                                        ),
                                    ]
                                )
                        elif isinstance(arg_value, (list, tuple)):
                            if arg_value:
                                benchmark_cmd.extend([arg_name] + [str(x) for x in arg_value])
                        else:
                            benchmark_cmd.extend([arg_name, str(arg_value)])

            except KeyError:
                pass

        print(benchmark_cmd)
        execute_process(benchmark_cmd)


if __name__ == "__main__":
    main()
