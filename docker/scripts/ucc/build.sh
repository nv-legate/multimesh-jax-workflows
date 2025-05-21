#! /usr/bin/env bash

set -e

pushd /opt/ucc

./autogen.sh
./configure --enable-mt --prefix=/usr \
  --with-cuda=/usr/local/cuda --with-java=no \
  --with-nvcc-gencode='-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90'

make -j
