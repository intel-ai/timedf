#!/bin/bash -e

. /usr/local/mapd-deps/mapd-deps.sh

cd build
make sanity_tests
