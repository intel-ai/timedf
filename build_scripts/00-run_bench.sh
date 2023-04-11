#!/bin/bash -e

mkdir -p ${PWD}/tmp

# ENV_NAME must be defined
if [[ -z "${ENV_NAME}" ]]; then
  echo "Please, provide ENV_NAME environment variable"
  exit 1
fi

if [[ -z "${PANDAS_MODE}" ]]; then
  echo "Please, provide PANDAS_MODE environment variable" 
  exit 1
else
  echo "PANDAS_MODE=${PANDAS_MODE}"
fi

# Run benchmark
conda run --live-stream -n $ENV_NAME benchmark-run $BENCH_NAME    \
                           -data_file "${DATA_FILE}"              \
                           -pandas_mode ${PANDAS_MODE}            \
                           -ray_tmpdir ${PWD}/tmp                 \
                           ${ADDITIONAL_OPTS}                     \
                           ${DB_COMMON_OPTS}                      \
                           "$@"
