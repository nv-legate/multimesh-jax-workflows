#! /usr/bin/env bash

set -e

pushd /opt/workspace/legate-jax

LEGATE_BUILD_TYPE=$1
BUILD_DIR=$2
REALM_DIR=$3
ZUKU_DIR=$4
LIB_DIR=$5
BAZEL_CACHE=$6
export CCACHE_DIR=/jax-plugin-ccache

mkdir -p /opt/lib/legate-jax

conda run --no-capture-out -n legere cmake -S . -B ${BUILD_DIR} \
  -DLegateJAX_RAPIDS_DIR=/opt/rapids-cmake \
  -DCPM_DOWNLOAD_LOCATION=/opt/cpm/cmake/CPM.cmake \
  -DCMAKE_GENERATOR:STRING="Unix Makefiles" \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE:STRING="${LEGATE_BUILD_TYPE}" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DLegateJAX_BAZEL_REMOTE_CACHE:STRING="${BAZEL_CACHE}" \
  -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCPM_xla_SOURCE:PATH=/opt/workspace/xla \
  -Dzuku_ROOT:PATH=${ZUKU_DIR} \
  -DLegion_ROOT:PATH=${REALM_DIR} \
  -Dzuku_SOURCE_DIR=/opt/workspace/zuku \
  -DLegateJAX_ASAN:BOOL=OFF \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

popd
