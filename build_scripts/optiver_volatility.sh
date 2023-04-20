#!/bin/bash -e

BENCH_NAME="optiver_volatility"
DATA_FILE="${DATASETS_PWD}/${BENCH_NAME}"

source $(dirname "$0")/00-run_bench.sh
