#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

apt-get update
apt install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
./llvm.sh 17

conda install -c conda-forge ninja ccache

popd
