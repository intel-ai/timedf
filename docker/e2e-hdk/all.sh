#!/bin/bash -e
# Set location to store datasets, 
# WARNING: don't store them in the same folder as dockerfile, to avoid long context loading
export DATASETS_ROOT=/localdisk/ekrivov/datasets
export RESULTS_DIR=results

mkdir -p ${DATASETS_ROOT}
mkdir -p ${RESULTS_DIR}

# Archive omniscripts for the upload 
tar -cf omniscripts.tar  --exclude=omniscripts.tar ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .

# Download data
./load_data.sh

# Run experiments
./run_docker.sh