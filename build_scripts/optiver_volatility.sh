#!/bin/bash -e

BENCH_NAME="optiver_volatility"
DATA_FILE="${DATASETS_PWD}/optiver_realized_volatility"

source $(dirname "$0")/00-run_bench.sh
