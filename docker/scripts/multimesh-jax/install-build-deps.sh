#! /usr/bin/env bash

set -e

pushd /opt/workspace/multimesh-jax

apt-get install -y --no-install-recommends wget
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.23.0/bazelisk-amd64.deb
dpkg -i bazelisk-amd64.deb
rm bazelisk-amd64.deb

apt-get update
apt install -y lsb-release wget software-properties-common gnupg git
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
./llvm.sh 17

conda install -c conda-forge ninja pybind11 ccache

popd
