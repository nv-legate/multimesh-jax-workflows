#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

export CCACHE_DIR=/ccache/zuku
BUILD_DIR=$1

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR}

popd
