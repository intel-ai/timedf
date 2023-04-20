#!/bin/bash -e

BENCH_NAME="ny_taxi"
DATASET_PATH="${DATASETS_PWD}/${BENCH_NAME}"

source $(dirname "$0")/00-run_bench.sh -dfiles_num 1
