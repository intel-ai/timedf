#!/bin/bash -e

BENCH_NAME="h2o"
DATA_FILE="${DATASETS_PWD}/h2o/J1_1e8_NA_0_0.csv" 

source $(dirname "$0")/00-run_bench.sh
