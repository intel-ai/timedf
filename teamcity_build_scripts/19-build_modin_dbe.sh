#!/bin/bash -e

python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check False --save_env True -task build                                 \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -executable "$PWD/../omniscidb/build/bin/omnisci_server"                                             \
                          --modin_path "$PWD/../modin/"
