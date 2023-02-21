#!/bin/sh

set -vxe

# HDK
cd /
tar -zxf build.tgz
cd hdk/build
make install
