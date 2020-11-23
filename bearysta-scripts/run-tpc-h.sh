#!/bin/bash
set -xe
# replacing string like fragment_size=5000000);"
sed -ri.bak -e "s/fragment_size=[0-9]+/fragment_size=$frags/" db-query-list-tpc-h.sql
numactl $numa -C $cpus $omnisci_server --config omnisci-bench-tpc-h.conf 2>&1
