import os
import time

from timedf.timer import tm
from timedf.arg_parser import parse_args, prepare_general_parser
from timedf.benchmark import BaseBenchmark, create_benchmark
from timedf.backend import Backend


def legacy_get_backend_params(args):
    """Returns backend_name and backend_params in new format."""
    backend = args.backend
    # Old system just passed pandas mode as backend, so we fix that
    if backend.startswith("Modin") or backend == "Pandas":
        pandas_mode = str(backend)
        print(pandas_mode)
        backend = "Pandas"

        params = {
            "pandas_mode": pandas_mode,
            "ray_tmpdir": args.ray_tmpdir,
            "ray_memory": args.ray_memory,
            # Used in ny_taxi_ml and plasticc
            "use_modin_xgb": args.use_modin_xgb,
            "num_threads": args.num_threads,
        }
        print(params)
    else:
        params = {"num_threads": args.num_threads}
    return backend, params


def legacy_remove_new_fields(params):
    # We want to add these 2 new fields and make them mandatory
    params = dict(params)
    params.pop("backend_name", None)
    params.pop("benchmark_name", None)
    return params


def make_benchmark() -> BaseBenchmark:
    """Function imports and creates benchmark object without using benchmark-specific arguments"""
    parser = prepare_general_parser()
    args = parser.parse_known_args()[0]
    bench_name = args.bench_name

    backend_name, backend_params = legacy_get_backend_params(args)
    # Set current backend, !!!needs to be run before benchmark import!!!
    Backend.init(backend_name, backend_params)
    return create_benchmark(bench_name)


def main():
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    benchmark = make_benchmark()
    args, db_config = parse_args(benchmark.add_benchmark_args)

    data_file = args.data_file.replace("'", "")
    # Set verbosity level for global object, shared with benchmarks
    tm.verbosity = args.verbosity

    run_parameters = {
        "data_file": data_file,
        "backend": args.backend,
        "no_ml": args.no_ml,
        "num_threads": args.num_threads,
        # Used in ny_taxi_ml and plasticc
        "use_modin_xgb": args.use_modin_xgb,
        # Add benchmark-specific arguments
        **{k: getattr(args, k) for k in benchmark.__params__},
    }

    report_args = {
        "tag": args.tag,
    }

    legacy_report = {
        "ray_tmpdir": args.ray_tmpdir,
        "ray_memory": args.ray_memory,
        # TODO: this parameter needs to be removed from run_params as well
        "use_modin_xgb": args.use_modin_xgb,
    }

    benchmarkDb = db_config.maybeCreateBenchmarkDb()

    run_id = int(round(time.time()))
    print(run_parameters)

    for iter_num in range(1, args.iterations + 1):
        print(f"Iteration #{iter_num}")
        results = benchmark.run(run_parameters)
        tm.reset()

        if benchmarkDb is not None:
            benchmarkDb.report(
                iteration_no=iter_num,
                name2time=results.measurements,
                params=results.params,
                benchmark=args.save_benchmark_name or args.bench_name,
                backend=args.save_backend_name or args.backend,
                run_id=run_id,
                run_params=legacy_remove_new_fields(
                    {**run_parameters, **report_args, **legacy_report}
                ),
            )


if __name__ == "__main__":
    main()
