#!/bin/bash -e

BENCH_NAME="ny_taxi"
DATASET_PATH="${DATASETS_PWD}/taxi"

source $(dirname "$0")/00-run_bench.sh -dfiles_num 1
