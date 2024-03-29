#!/bin/bash -eu
if [ -z ${DATASETS_PATH+x} ]; then
    echo "Please, set the env variable \"DATASETS_PATH\"!";
    exit 1
else
    echo "DATSETS_ROOT=$DATASETS_PATH"
fi

mkdir -p ${DATASETS_PATH}
chmod 0777 ${DATASETS_PATH}

USER_ID="$(id -u):$(id -g)"

# awsd="docker run --rm -it -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} --user ${USER_ID} -v ${DATASETS_PATH}:/datasets amazon/aws-cli s3"
awsd="docker run --rm -it  --user ${USER_ID} -v ${DATASETS_PATH}:/datasets amazon/aws-cli s3"

$awsd sync s3://modin-datasets/taxi /datasets/taxi --no-sign-request --exclude "*" --include "trips_xa[abcdefghijklmnopqrst].csv"
$awsd sync s3://modin-datasets/plasticc /datasets/plasticc --no-sign-request --exclude "*" --include "*.csv"
$awsd sync s3://modin-datasets/census /datasets/census --no-sign-request
