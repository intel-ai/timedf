#!/bin/bash -e

BENCH_NAME="ny_taxi_ml"
DATA_FILE="${DATASETS_PWD}/yellow-taxi-dataset/"

if [ "$PANDAS_MODE" = "Modin_on_ray" ]; then
    USE_MODIN_XGB="-use_modin_xgb"
else
    USE_MODIN_XGB=""
fi

source $(dirname "$0")/00-run_bench.sh "${USE_MODIN_XGB}"
