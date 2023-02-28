#!/bin/bash -e

BENCH_NAME="ny_taxi"
DATA_FILE="${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv"

source $(dirname "$0")/00-run_bench.sh -dfiles_num 1
