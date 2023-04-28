import argparse
from pathlib import Path

from omniscripts.benchmark import create_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset for benchmark")
    parser.add_argument("benchmark", help="Benchmark name")
    parser.add_argument(
        "target_dir", help="Folder where dataset will be stored. This folder will be created."
    )
    parser.add_argument(
        "-r", "--reload", default=False, action="store_true", help="Rewrite existing files."
    )
    return parser.parse_args()


def load_dataset(benchmark_name, target_dir, reload):
    benchmark = create_benchmark(benchmark_name)
    benchmark.load_data(target_dir=Path(target_dir), reload=reload)


def main():
    args = parse_args()
    load_dataset(args.benchmark, args.target_dir, args.reload)


if __name__ == "__main__":
    main()
