#! /usr/bin/env bash

pushd /opt/ucx

./autogen.sh
./contrib/configure-release --enable-mt \
  --with-cuda=/usr/local/cuda --with-java=no

make -j
