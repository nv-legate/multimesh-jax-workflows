#! /usr/bin/env bash

apt-get update

apt-get install -y --no-install-recommends wget
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.23.0/bazelisk-amd64.deb
dpkg -i bazelisk-amd64.deb
rm bazelisk-amd64.deb

apt install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
./llvm.sh 17
