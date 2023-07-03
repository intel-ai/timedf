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
if [ ${PANDAS_MODE} = "Modin_on_unidist_mpi" ]; then
    conda run --live-stream -n $ENV_NAME mpiexec -n 1 benchmark-run $BENCH_NAME    \
                           -data_file "${DATASETS_PWD}/${BENCH_NAME}" \
                           -backend ${PANDAS_MODE}            \
                           ${ADDITIONAL_OPTS}                     \
                           ${DB_COMMON_OPTS}                      \
                           "$@"
else
  conda run --live-stream -n $ENV_NAME benchmark-run $BENCH_NAME    \
                           -data_file "${DATASETS_PWD}/${BENCH_NAME}" \
                           -backend ${PANDAS_MODE}            \
                           ${ADDITIONAL_OPTS}                     \
                           ${DB_COMMON_OPTS}                      \
                           "$@"

fi

