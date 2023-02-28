#!/bin/bash -e

BENCH_NAME="ny_taxi_ml"
DATA_FILE="${DATASETS_PWD}/yellow-taxi-dataset/"

if [ "$PANDAS_MODE" = "Modin_on_ray" ]; then
    USE_MODIN_XGB="True"
else
    USE_MODIN_XGB="False"
fi

source $(dirname "$0")/00-run_bench.sh -use_modin_xgb "${USE_MODIN_XGB}"
