#! /usr/bin/env bash

apt-get install -y --no-install-recommends wget curl

for arg in "$@"; do
 case "$arg" in
    nsys) nsys=1 ;;
    jupyter) jupyter=1 ;;
    *) echo "Unknown component: $arg" ;;
  esac
done

mkdir -p /opt/tools

pushd /opt/tools
if [ ! -z $nsys ]; then
  wget ${NSYS_URL}
  curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub > nvidia.pub
fi
popd
