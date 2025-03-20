#! /usr/bin/env bash

set -e

pushd /opt/workspace/legate-jax

export CCACHE_DIR=/ccache/legate-jax
BUILD_DIR=$1

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR} --parallel

popd
