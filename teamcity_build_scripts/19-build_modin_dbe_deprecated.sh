#!/bin/bash -e

python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check False --save_env True -task build                                \
                          -executable "$PWD/../omniscidb/build/bin/omnisci_server"                                             \
                          --modin_path "$PWD/../modin/"
