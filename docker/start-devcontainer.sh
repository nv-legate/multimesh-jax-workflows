#! /usr/bin/env bash

docker run -it --cap-add SYS_ADMIN \
    --entrypoint /bin/bash \
    --net=host \
    --add-host=host.docker.internal:host-gateway \
    --mount type=bind,source="$(pwd)"/mount,target=/mnt \
    --mount type=bind,source=$(realpath ../.git),target=/.git \
    --mount type=bind,source=$HOME/.gitconfig,target=/root/.gitconfig \
    -v ~/.ssh:/root/.ssh:ro \
    -d \
    --gpus 2 \
    --name $2 \
    $1

