#!/bin/bash -e

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

# ENV_NAME must be provided
# live stream will provide live stdout and stderr
conda run --live-stream -n $ENV_NAME benchmark-run $BENCH_NAME    \
                           -data_file "${DATASETS_PWD}/${BENCH_NAME}" \
                           -backend ${PANDAS_MODE}            \
                           ${ADDITIONAL_OPTS}                     \
                           ${DB_COMMON_OPTS}                      \
                           "$@"
