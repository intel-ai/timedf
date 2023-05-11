#!/bin/bash -e

####################
# Argument parsing #

usage() { echo "Usage: $0 -d DATASETS_PATH [-r xlsx]" 1>&2; exit 1; }

REPORT=""
DB_COMMON_OPTS=""
while getopts ":d:r:" o; do
    case "${o}" in
        d)
            DATASETS_PATH=${OPTARG}
            ;;
        r)
            REPORT=${OPTARG}
            [[ "$REPORT" == "xlsx" ]] || ( echo "Unrecognized report type, only xlsx is supported" && usage )
            DB_COMMON_OPTS="-db_name /results/result_database.sqlite"
            ;;
        *)
            echo "Unrecognized argument ${OPTARG}"
            usage
            ;;
    esac
done

[ -z "$DATASETS_PATH" ] && echo "Please, provide DATASETS_PATH" && usage

#
####################

# Set location to store datasets, 
# WARNING: paths need to be absolute, otherwise docker will not mount. The `($readlink -m ...)` part will make provided path absolute, so use it if necessary
# WARNING: don't store datasets in the same folder as dockerfile, to avoid long context loading during docker build
export DATASETS_PATH=$(readlink -m $DATASETS_PATH)
export RESULTS_DIR=$(readlink -m results)
export REPORT
export DB_COMMON_OPTS

echo "Configured parameters: DATSETS_ROOT=$DATASETS_PATH, RESULTS_DIR=$RESULTS_DIR, REPORT=$REPORT"

# Archive omniscripts for the upload 
tar -cf omniscripts.tar  --exclude=e2e-hdk ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
echo "Building docker image"
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .

# Download data
./load_data.sh

# Run experiments & generate xlsx if -r xlsx was used
./run_docker.sh
