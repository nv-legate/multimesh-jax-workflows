#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

export CCACHE_DIR=/zuku-ccache
BUILD_TYPE=$1
BUILD_DIR=/opt/build/zuku
LIB_DIR=/opt/lib
REALM_BUILD_DIR=${2:-}

if [ -d /opt/build/realm ]; then
  realm_dir_flag="-DLegion_ROOT=/opt/build/realm"
fi

conda run --no-capture-out -n legere \
  cmake -S . -B ${BUILD_DIR} \
  -Dzuku_RAPIDS_DIR=/opt/rapids-cmake \
  -DCPM_DOWNLOAD_LOCATION=/opt/cpm/cmake/CPM.cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
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
  $realm_dir_flag \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

ln -s $BUILD_DIR/compile_commands.json

popd
