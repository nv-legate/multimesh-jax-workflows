#! /usr/bin/env bash

set -e

editable_flag=${1:-}
BUILD_DIR=/opt/build/multimesh-jax

pushd /opt/workspace/multimesh-jax
export CMAKE_ARGS="-DCPM_xla_SOURCE:PATH=/opt/workspace/xla -DMultiMeshJAX_ROOT:PATH=${BUILD_DIR} -DMultiMeshJAX_RAPIDS_DIR=/opt/rapids-cmake -DCPM_DOWNLOAD_LOCATION=/opt/cpm/cmake/CPM.cmake"
export SETUPTOOLS_ENABLE_FEATURES=legacy-editable
conda run -n legere cmake --install ${BUILD_DIR}
conda run --no-capture-out -n legere python -m pip install $editable_flag .
popd
