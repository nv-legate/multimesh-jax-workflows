#! /usr/bin/env bash

apt-get update

apt-get install -y --no-install-recommends \
  gcc-12 g++-12 apt-transport-https autoconf automake \
  libtool build-essential devscripts debhelper fakeroot \
  python3

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
