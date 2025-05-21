#! /usr/bin/env bash

set -e

pushd /opt/workspace/multimesh-jax

BUILD_TYPE=$1
BAZEL_CACHE=$2
BUILD_TESTS=${3:-OFF}
BUILD_DIR=/opt/build/multimesh-jax
LIB_DIR=/opt/lib
export CCACHE_DIR=/jax-plugin-ccache

mkdir -p /opt/lib/multimesh-jax

if [ -d /opt/build/zuku ]; then
  zuku_dir_flag="-Dzuku_ROOT=/opt/build/zuku"
fi

if [ -d /opt/build/realm ]; then
  realm_dir_flag="-DLegion_ROOT=/opt/build/realm"
fi


conda run --no-capture-out -n legere cmake -S . -B ${BUILD_DIR} \
  -DMultiMeshJAX_ENABLE_TESTS=${BUILD_TESTS} \
  -DMultiMeshJAX_RAPIDS_DIR=/opt/rapids-cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCPM_DOWNLOAD_LOCATION=/opt/cpm/cmake/CPM.cmake \
  -DCMAKE_GENERATOR:STRING="Unix Makefiles" \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${LIB_DIR} \
  -DMultiMeshJAX_BAZEL_REMOTE_CACHE:STRING="${BAZEL_CACHE}" \
  -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCPM_xla_SOURCE:PATH=/opt/workspace/xla \
  -Dzuku_SOURCE_DIR=/opt/workspace/zuku \
  -DMultiMeshJAX_ASAN:BOOL=OFF \
  $zuku_dir_flag \
  $realm_dir_flag \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

ln -s $BUILD_DIR/compile_commands.json

popd
