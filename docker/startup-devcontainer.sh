#! /usr/bin/bash

BAZEL_CACHE=grpc://host.docker.internal:9092
BUILD_TYPE=${1:-Release}
BUILD_DIR=${2:-/opt/build}
LIB_DIR=${3:-/opt/lib}

# map the git submodule worktrees to the correct path
mkdir -p /docker
ln -s /opt/workspace /docker/workspace

# git finds what I'm doing 'dubious'
git config --global --add safe.directory /opt/workspace/xla
git config --global --add safe.directory /opt/workspace/multimesh-jax
git config --global --add safe.directory /opt/workspace/realm
git config --global --add safe.directory /opt/workspace/zuku

pushd /opt/workspace/realm
/opt/configure-realm.sh ${BUILD_TYPE} ${BUILD_DIR}/realm ${LIB_DIR}
/opt/build-realm.sh ${BUILD_DIR}/realm
popd

pushd /opt/workspace/zuku
/opt/configure-zuku.sh ${BUILD_TYPE} ${BUILD_DIR}/zuku ${BUILD_DIR}/realm ${LIB_DIR}
/opt/build-zuku.sh ${BUILD_DIR}/zuku
popd

pushd /opt/workspace/xla
/opt/configure-xla.sh
popd

pushd /opt/workspace/multimesh-jax
/opt/configure-multimesh-jax.sh ${BUILD_TYPE} ${BUILD_DIR}/multimesh-jax ${BUILD_DIR}/realm ${BUILD_DIR}/zuku ${LIB_DIR} ${BAZEL_CACHE}
/opt/build-multimesh-jax.sh ${BUILD_DIR}/multimesh-jax
/opt/install-multimesh-jax.sh ${BUILD_DIR}/multimesh-jax
popd

pushd /opt/workspace/xla
#/opt/xla-compile-commands.sh
popd
