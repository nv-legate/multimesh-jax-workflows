#! /usr/bin/env bash

apt remove -y --allow-change-held-packages libnccl2 libnccl-dev
apt install -y --reinstall ./build/pkg/deb/libnccl2*.deb
apt -y install --reinstall ./build/pkg/deb/libnccl-dev*.deb
