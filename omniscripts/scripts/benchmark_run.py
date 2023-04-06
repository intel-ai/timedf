import os
import time

from omniscripts.arg_parser import parse_args, prepare_general_parser
from omniscripts.benchmark import BaseBenchmark
from omniscripts.benchmarks import create_benchmark
from omniscripts.pandas_backend import Backend


def make_benchmark() -> BaseBenchmark:
    """Function imports and creates benchmark object without using benchmark-specific arguments"""
    parser = prepare_general_parser()
    args = parser.parse_known_args()[0]
    bench_name = args.bench_name

    # Set current backend, !!!needs to be run before benchmark import!!!
    Backend.init(
        backend_name=args.pandas_mode, ray_tmpdir=args.ray_tmpdir, ray_memory=args.ray_memory
    )

    return create_benchmark(bench_name)


def main():
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    benchmark = make_benchmark()
    args, db_config = parse_args(benchmark.add_benchmark_args)

    data_file = args.data_file.replace("'", "")

    run_parameters = {
        "data_file": data_file,
        "pandas_mode": args.pandas_mode,
        "no_ml": args.no_ml,
        # used only in ny_taxi
        "dfiles_num": args.dfiles_num,
        # Used in ny_taxi_ml and plasticc
        "use_modin_xgb": args.use_modin_xgb,
        # Used only for census
        "optimizer": args.optimizer,
        # Used only for census
        "gpu_memory": args.gpu_memory,
        # Used only for old H2O
        "extended_functionality": args.extended_functionality,
        # Add benchmark-specific arguments
        **{k: getattr(args, k) for k in benchmark.__params__},
    }

    report_args = {
        "ray_tmpdir": args.ray_tmpdir,
        "ray_memory": args.ray_memory,
        "commit_hdk": args.commit_hdk,
        "commit_omniscripts": args.commit_omniscripts,
        "commit_modin": args.commit_modin,
    }

    if db_config is not None:
        from ..report import BenchmarkDb

        reporter = BenchmarkDb(db_config.create_engine())
    else:
        reporter = None

    run_id = int(round(time.time()))
    print(run_parameters)

    for iter_num in range(1, args.iterations + 1):
        print(f"Iteration #{iter_num}")
        results = benchmark.run(run_parameters)

        if reporter is not None:
            reporter.report(
                iteration_no=iter_num,
                name2time=results.measurements,
                params=results.params,
                benchmark=args.bench_name,
                run_id=run_id,
                run_params={**run_parameters, **report_args},
            )


if __name__ == "__main__":
    main()
