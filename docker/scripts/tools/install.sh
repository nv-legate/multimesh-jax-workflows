#! /usr/bin/env bash

apt-get update

apt-get install numactl

for arg in "$@"; do
 case "$arg" in
    nsys) nsys=1 ;;
    jupyter) jupyter=1 ;;
    *) echo "Unknown component: $arg" ;;
  esac
done

if [ ! -z $nsys ]; then
  pushd opt
  curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | apt-key add -
  echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/nsys.list

  wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsight-systems-2024.1.1_2024.1.1.59-1_amd64.deb
  apt-get update
  dpkg --configure -a
  apt-get -y -f install
  apt-get -y install nsight-systems-2024.1.1 --reinstall

  mkdir -p /root/.config/NVIDIA\ Corporation
  echo "CuptiUsePerThreadBuffer=false" > /root/.config/NVIDIA\ Corporation/nsys-config.ini
fi

if [ ! -z $jupyter ]; then
  conda run -n legere --no-capture-output python -m pip install notebook --no-cache-dir
fi
