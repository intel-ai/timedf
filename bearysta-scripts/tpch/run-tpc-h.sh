#!/bin/bash
set -xe
bash prepare_tables.sh $scale $OMP_NUM_THREADS
numactl $numa -C $cpus $omnisci_server --config omnisci-bench-tpc-h.conf --data generated/data-$scale-$OMP_NUM_THREADS 2>&1
