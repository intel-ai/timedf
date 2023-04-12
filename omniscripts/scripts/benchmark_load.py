import argparse
from pathlib import Path

from omniscripts.benchmarks import create_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset for benchmark")
    parser.add_argument("benchmark", help="Benchmark name")
    parser.add_argument(
        "dataset_dir", help="Directory where all datasets are stored. New folder will be created."
    )
    parser.add_argument(
        "-r", "--reload", default=False, action="store_true", help="Rewrite existing files."
    )
    return parser.parse_args()


def load_dataset(benchmark_name, dataset_dir, reload):
    benchmark = create_benchmark(benchmark_name)
    target_dir = Path(dataset_dir) / benchmark_name
    benchmark.load_data(target_dir=target_dir, reload=reload)


def main():
    args = parse_args()
    load_dataset(args.benchmark, args.dataset_dir, args.reload)


if __name__ == "__main__":
    main()
