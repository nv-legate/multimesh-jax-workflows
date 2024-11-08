#! /usr/bin/env bash

mkdir -p ~/.cache/ccache/zuku
mkdir -p ~/.cache/ccache/realm
mkdir -p ~/.cache/ccache/multimesh-jax

docker run -it --cap-add SYS_ADMIN \
    --entrypoint /bin/bash \
    --net=host \
    --add-host=host.docker.internal:host-gateway \
    --mount type=bind,source=$(pwd)/workspace/jax,target=/opt/workspace/jax \
    --mount type=bind,source=$(pwd)/workspace/xla,target=/opt/workspace/xla \
    --mount type=bind,source=$(pwd)/workspace/multimesh-jax,target=/opt/workspace/multimesh-jax \
    --mount type=bind,source=$(pwd)/workspace/realm,target=/opt/workspace/realm \
    --mount type=bind,source=$(pwd)/workspace/zuku,target=/opt/workspace/zuku \
    --mount type=bind,source=$(pwd)/target,target=/target \
    --mount type=bind,source=$HOME/.gitconfig,target=/root/.gitconfig \
    --mount type=bind,source=$(pwd)/workspace/build,target=/opt/build \
    --mount type=bind,source=$HOME/.cache/ccache/realm,target=/ccache/realm \
    --mount type=bind,source=$HOME/.cache/ccache/zuku,target=/ccache/zuku \
    --mount type=bind,source=$HOME/.cache/ccache/multimesh-jax,target=/ccache/multimesh-jax \
    --mount type=bind,source=$(pwd)/workspace/bazel,target=/root/.cache/bazel \
    -v ~/.ssh:/root/.ssh:ro \
    --gpus 2 \
    --entrypoint /bin/bash \
    $1

