#!/bin/bash -e

mkdir -p ${PWD}/tmp


# ENV_NAME must be provided
if [[ -z "${PANDAS_MODE}" ]]; then
  echo "Please, provide PANDAS_MODE environment variable" 
  exit 1
else
  echo "PANDAS_MODE=${PANDAS_MODE}"
fi

# Run benchmark
conda run -n $ENV_NAME python3 run_modin_tests.py -task benchmark                      \
                           -use_modin_xgb ${USE_MODIN_XGB}      \
                           -bench_name $BENCH_NAME              \
                           -data_file "${DATA_FILE}"            \
                           -pandas_mode ${PANDAS_MODE}          \
                           -ray_tmpdir ${PWD}/tmp               \
                           ${ADDITIONAL_OPTS}                   \
                           ${ADDITIONAL_OPTS_NIGHTLY}           \
                           ${DB_COMMON_OPTS}                    \
                           "$@"
