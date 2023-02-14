#!/bin/bash -e
cd omniscripts

export ADDITIONAL_OPTS="-iterations 3"

# Modin_on_hdk
./teamcity_build_scripts/33-ny_taxi_modin_on_hdk_20M_records.sh |& tee /results/taxi_hdk.res
./teamcity_build_scripts/30-census_modin_on_hdk.sh |& tee /results/census_hdk.res
./teamcity_build_scripts/34-plasticc_modin_on_hdk.sh |& tee /results/plasticc_hdk.res

# Stock pandas
# HDK scripts deactivate conda, so we need to reactivate it again
source ${CONDA_PREFIX}/bin/activate
./teamcity_build_scripts/42-ny_taxi_pandas_20M_records.sh |& tee /results/taxi_pandas.res
./teamcity_build_scripts/41-census_pandas.sh |& tee /results/census_pandas.res
./teamcity_build_scripts/43-plasticc_pandas.sh |& tee /results/plasticc_pandas.res

# HDK scripts deactivate conda, so we need to reactivate it again

case $REPORT in

  xlsx)
    echo -n "Generating xlsx report"
    # We need to activate env to have all the libraries for report generation
    source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}
    PYTHONPATH=./ python3 ./scripts/generate_report.py -report_path /results/report.xlsx -db_name /results/result_database.sqlite -agg median
    ;;

  *)
    echo -n "No report generation"
    ;;
esac
