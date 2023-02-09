#!/bin/bash -eu
# Set location to store datasets, 
# WARNING: paths need to be absolute, otherwise docker will not mount. The `($readlink -m ...)` part will make provided path absolute, so use it if necessary
# WARNING: don't store datasets in the same folder as dockerfile, to avoid long context loading during docker build
export DATASETS_ROOT=$(readlink -m $DATASETS_ROOT)
export RESULTS_DIR=$(readlink -m results)

# Archive omniscripts for the upload 
tar -cf omniscripts.tar  --exclude=e2e-hdk ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
echo "Building docker image"
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .

# Download data
./load_data.sh

# Run experiments & generate xlsx
./run_docker.sh
