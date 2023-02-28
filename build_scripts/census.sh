#!/bin/bash -e


BENCH_NAME="census"
DATA_FILE="${DATASETS_PWD}/census/ipums_education2income_1970-2010.csv.gz" 

source $(dirname "$0")/00-run_bench.sh
