#! /usr/bin/env bash

set -e

pushd /opt/workspace/multimesh-jax

apt-get update
apt-get install -y --no-install-recommends wget
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.23.0/bazelisk-amd64.deb
dpkg -i bazelisk-amd64.deb
rm bazelisk-amd64.deb

apt-get install -y lsb-release wget software-properties-common gnupg
wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main' -y
apt-get update
apt-get install -y clang-17 lld-17 --no-install-recommends

ln -s /usr/bin/lld-17 /usr/bin/lld

conda run -n legere conda install -c conda-forge ninja pybind11 ccache

popd
