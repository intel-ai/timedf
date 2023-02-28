#!/bin/bash -e

BENCH_NAME="h2o"
DATA_FILE="${DATASETS_PWD}/h2o/G1_1e9_1e2_0_0.csv"

source $(dirname "$0")/00-run_bench.sh
