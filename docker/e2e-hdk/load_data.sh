#!/bin/bash -e
mkdir -p ${DATASETS_ROOT}
chmod 0777 ${DATASETS_ROOT}

aws s3 sync s3://modin-datasets/taxi ${DATASETS_ROOT}/taxi --no-sign-request --exclude "*" --include "trips_xa*.csv"
aws s3 sync s3://modin-datasets/plasticc ${DATASETS_ROOT}/plasticc --no-sign-request --exclude "*" --include "*.csv"
aws s3 sync s3://modin-datasets/census ${DATASETS_ROOT}/census --no-sign-request
