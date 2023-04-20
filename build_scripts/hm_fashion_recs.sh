#!/bin/bash -e

BENCH_NAME="hm_fashion_recs"
DATA_FILE="${DATASETS_PWD}/${BENCH_NAME}"

source $(dirname "$0")/00-run_bench.sh
