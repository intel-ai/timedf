#!/bin/bash -e
eval source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}

# Basic config
export MODIN_EXPERIMENTAL=true
export MODIN_TEST_DATASET_SIZE=big
export MODIN_ASV_DATASIZE_CONFIG=`pwd`/omniscripts/docker/microbenchmarks-hdk/modin-asv-datasize-config.json

cd modin/asv_bench

# setup ASV machines
asv machine --yes
asv machine --machine ${HOST_NAME}-ray --yes
asv machine --machine ${HOST_NAME}-hdk --yes
asv machine --machine ${HOST_NAME}-pandas --yes

RAY_MACHINE_NAME=$HOST_NAME-ray
HDK_MACHINE_NAME=$HOST_NAME-hdk
PANDAS_MACHINE_NAME=$HOST_NAME-pandas

###############
# Ray config ##
###############

export MODIN_ENGINE=ray

asv run --launch-method=spawn --config asv.conf.json \
    -b ^hdk --machine $RAY_MACHINE_NAME -a repeat=3 -a number=1 \
    --show-stderr --python=same --set-commit-hash HEAD
RAY_RESULT_NAME=`ls .asv/results/$RAY_MACHINE_NAME/ | grep existing`


##############
# HDK config #
##############

export MODIN_ENGINE=native
export MODIN_STORAGE_FORMAT=hdk

# 'number=1' allows for each bench run to get tables with different names (in `trigger_import`),
# which eliminates caching inside HDK (`setup` is called before every run)
asv run --launch-method=forkserver --config asv.conf.hdk.json \
    -b ^hdk --machine $HDK_MACHINE_NAME -a repeat=3 -a number=1 \
    --show-stderr --python=same --set-commit-hash HEAD
HDK_RESULT_NAME=`ls .asv/results/$HDK_MACHINE_NAME/ | grep existing`

################# 
# Pandas config #
#################

export MODIN_ASV_USE_IMPL=pandas

asv run --launch-method=forkserver --config asv.conf.hdk.json \
    -b ^hdk --machine $PANDAS_MACHINE_NAME -a repeat=3 \
    --show-stderr --python=same --set-commit-hash HEAD
PANDAS_RESULT_NAME=`ls .asv/results/$PANDAS_MACHINE_NAME/ | grep existing`

# report modin on hdk results
if [ -z "$DB_COMMON_OPTS" ]; then
    echo env variable DB_COMMON_OPTS is undefined - no results will be report
    exit 0
fi

cd ../../omniscripts
python report_asv_result.py \
    --result-path ../modin/asv_bench/.asv/results/$RAY_MACHINE_NAME/$RAY_RESULT_NAME \
    $DB_COMMON_OPTS

python report_asv_result.py \
    --result-path ../modin/asv_bench/.asv/results/$HDK_MACHINE_NAME/$HDK_RESULT_NAME \
    $DB_COMMON_OPTS


# report pandas results
python report_asv_result.py \
    --result-path ../modin/asv_bench/.asv/results/$PANDAS_MACHINE_NAME/$PANDAS_RESULT_NAME \
    $DB_COMMON_OPTS
