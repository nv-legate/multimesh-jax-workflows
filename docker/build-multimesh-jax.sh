#! /usr/bin/env bash

set -e

pushd /opt/workspace/multimesh-jax

export CCACHE_DIR=/ccache/multimesh-jax
BUILD_DIR=$1

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR} --parallel

popd
