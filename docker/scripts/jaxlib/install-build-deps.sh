#! /usr/bin/env bash

apt-get install -y lsb-release wget software-properties-common gnupg --no-install-recommends

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.23.0/bazelisk-amd64.deb
dpkg -i bazelisk-amd64.deb
rm bazelisk-amd64.deb

wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main' -y
apt-get update
apt-get install -y clang-17 --no-install-recommends
