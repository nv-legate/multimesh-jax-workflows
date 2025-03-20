#! /usr/bin/env bash

set -e

BUILD_DIR=$1
editable_flag=${2:-}

pushd /opt/workspace/legate-jax
export CMAKE_ARGS="-DCPM_xla_SOURCE:PATH=/opt/workspace/xla -DLegateJAX_ROOT:PATH=${BUILD_DIR}"
export SETUPTOOLS_ENABLE_FEATURES=legacy-editable
#conda run --no-capture-out -n legere cmake --install ${BUILD_DIR} --prefix /opt/install/miniconda/envs/legere
conda run --no-capture-out -n legere python -m pip install $editable_flag .
popd
