import os

from omniscripts import run_benchmarks
from omniscripts.arg_parser import DbConfig, prepare_parser


def main(raw_args=None):
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = prepare_parser()
    args = parser.parse_args(raw_args)

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


if __name__ == "__main__":
    main()
