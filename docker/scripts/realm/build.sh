#! /usr/bin/env bash

set -e

pushd /opt/workspace/realm

BUILD_DIR=/opt/build/realm
export CCACHE_DIR=/ccache/realm

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR}

popd
