#! /usr/bin/env bash

apt-get install -y --no-install-recommends numactl

for arg in "$@"; do
 case "$arg" in
    nsys) nsys=1 ;;
    jupyter) jupyter=1 ;;
    *) echo "Unknown component: $arg" ;;
  esac
done

pushd /opt/tools
if [ ! -z $nsys ]; then
  if [ -z "$NSYS_DEB" ]; then
    echo "Error: NSYS_DEB is empty or not set"
    exit 1
  fi
  cat nvidia.pub | apt-key add -
  echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/nsys.list

  apt-get update
  dpkg --configure -a
  apt-get -y -f install --no-install-recommends
  apt-get -y install ./${NSYS_DEB} --reinstall --no-install-recommends

  mkdir -p /root/.config/NVIDIA\ Corporation
  echo "CuptiUsePerThreadBuffer=false" > /root/.config/NVIDIA\ Corporation/nsys-config.ini
fi

if [ ! -z $jupyter ]; then
  conda run -n legere --no-capture-output python -m pip install notebook --no-cache-dir
fi

popd
