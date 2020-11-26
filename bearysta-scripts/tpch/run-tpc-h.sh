#!/bin/bash
set -xe
numactl $numa -C $cpus $omnisci_server --config omnisci-bench-tpc-h.conf 2>&1
