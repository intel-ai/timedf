#!/bin/bash -e

# This variable is used to improve performance and its value was obtained during the experiment.
# Each workload can have a different value.
export MODIN_HDK_FRAGMENT_SIZE=32000000

BENCH_NAME="plasticc"
DATASET_PATH="${DATASETS_PWD}/${BENCH_NAME}"

# This benchmark also support USE_MODIN_XGB="-use_modin_xgb"
USE_MODIN_XGB=""
source $(dirname "$0")/00-run_bench.sh ${USE_MODIN_XGB}
