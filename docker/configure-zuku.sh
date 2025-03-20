#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

export CCACHE_DIR=/zuku-ccache
BUILD_TYPE=$1
BUILD_DIR=$2
REALM_DIR=$3
LIB_DIR=$4

conda run --no-capture-out -n legere \
  cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_GENERATOR:STRING=Ninja \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DBUILD_SHARED_LIBS:BOOL=ON \
  -DLegion_ROOT:PATH=${REALM_DIR} \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

popd
