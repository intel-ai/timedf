#!/bin/bash -xe
set -o pipefail

# to resolve problem with proxy in docker for ray:
# https://github.com/modin-project/modin/issues/2608
PIP_CONF_PATH=pip.conf
cat << EOF > $PIP_CONF_PATH
[global]
proxy = `echo $http_proxy`
EOF
mkdir $HOME/.pip 
cp $PIP_CONF_PATH $HOME/.pip/
unset http_proxy
unset https_proxy

# setup git client
git config --global user.name teamcity
git config --global user.email teamcity@teamcity.com

# describe machine for asv - this is needed due to an issue in current asv version
MACHINE_INFO_PATH=$HOME/.asv-machine.json
cat << EOF > $MACHINE_INFO_PATH
{
    "version": 1,
    "xeon-e5": {
        "arch": "x86_64",
        "cpu": "Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz",
        "machine": "xeon-e5",
        "num_cpu": "44",
        "os": "Linux 5.4.0-54-generic",
        "ram": "131910328"
    },
    "xeon-e5-pandas": {
        "arch": "x86_64",
        "cpu": "Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz",
        "machine": "xeon-e5-pandas",
        "num_cpu": "44",
        "os": "Linux 5.4.0-54-generic",
        "ram": "131910328"
    }
}
EOF

# setup for results storing - pls make sure token is not disclosed
MODIN_BENCH_RESULTS_PATH="./modin-bench"
git clone https://anmyachev:$AMYACHEV_GITHUB_TOKEN@github.com/modin-project/modin-bench.git

mkdir asv_bench/.asv
# copy the previous results to the folder where the new results will be added
# so that potential conflicts can be resolved by ASV itself
cp -r $MODIN_BENCH_RESULTS_PATH/results/ asv_bench/.asv/
cd asv_bench

HASHFILE_PATH=./hashfile.txt
echo `git rev-parse HEAD` > $HASHFILE_PATH
#echo 3fb4f7612bba90605c1d553809eb367312d43f2e > $HASHFILE_PATH
#echo 50457cc21eef4e1f6039657b54037b4178f84144 >> $HASHFILE_PATH

conda config --set channel_priority flexible

if [[ -f asv.conf.hdk.json ]];
then
    # benchmarking HDK backend
    # 'number=1' allows for each bench run to get tables with different names (in `trigger_import`),
    # which eliminates caching inside HDK (`setup` is called before every run)
    MODIN_ENGINE=native MODIN_STORAGE_FORMAT=hdk MODIN_EXPERIMENTAL=true asv run HASHFILE:hashfile.txt \
        --show-stderr --machine xeon-e5 --launch-method=forkserver \
       --config asv.conf.hdk.json -b ^hdk -a number=1
fi

# benchmarking Modin on Ray
asv run HASHFILE:hashfile.txt --show-stderr --machine xeon-e5 --launch-method=spawn \
    -b ^benchmarks -b ^io -b ^scalability

# benchmarking pure pandas
MODIN_ASV_USE_IMPL=pandas asv run HASHFILE:hashfile.txt --show-stderr --machine xeon-e5-pandas \
    --launch-method=spawn -b ^benchmarks -b ^io

cd ..

cp -r asv_bench/.asv/results $MODIN_BENCH_RESULTS_PATH
cd $MODIN_BENCH_RESULTS_PATH
git add results/
git status
git commit -m 'New results'
git push

