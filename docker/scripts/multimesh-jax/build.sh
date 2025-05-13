#! /usr/bin/env bash

set -e

pushd /opt/workspace/multimesh-jax

export CCACHE_DIR=/ccache/multimesh-jax
BUILD_DIR=/opt/build/multimesh-jax

conda run --no-capture-out -n legere cmake --build ${BUILD_DIR} --parallel

popd
