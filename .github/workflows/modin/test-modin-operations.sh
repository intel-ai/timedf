#!/bin/sh

set -vxe
cd /_work

# HDK
(
    mkdir hdk
    cd hdk
    tar -zxf /_work/release.tgz -C $CONDA_PREFIX install-prefix
)

# Modin installation
(
    git clone https://github.com/modin-project/modin.git
    cd modin && pip install -e .[ray]

    git apply /_work/omniscripts/.github/workflows/modin/groupby.patch
)

# ASV installation
(
    git clone https://github.com/airspeed-velocity/asv.git
    cd asv
    git checkout ef016e233cb9a0b19d517135104f49e0a3c380e9
    git apply ../omniscripts/docker/microbenchmarks-hdk/asv-default-timeout.patch
    cd ..
    cd asv && pip install -e .
)

echo "asv script launch"
export ENV_NAME=modin-test
export HOST_NAME=c9n7
export MODIN_USE_CALCITE=True

bash -vx ./omniscripts/docker/microbenchmarks-hdk/asv-runner.sh

