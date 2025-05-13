#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

export CCACHE_DIR=/ccache/zuku
BUILD_DIR=/opt/build/zuku

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR}

popd
