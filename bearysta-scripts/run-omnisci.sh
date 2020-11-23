#!/bin/bash
set -xe
# replacing string like "storage_type='CSV:trips-header.csv', fragment_size=5000000);"
sed -ri.bak -e "s/trips.+\.csv/trips-header-$rows.csv/; s/fragment_size=[0-9]+/fragment_size=$frags/" db-query-list-taxi.sql
numactl $numa -C $cpus $omnisci_server --config omnisci-bench-taxi.conf 2>&1
