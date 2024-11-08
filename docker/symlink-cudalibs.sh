#! /usr/bin/env bash

# symlink cudnn into a private directory
# to avoid include conflicts
mkdir -p /opt/nvidia/cudnn
pushd /opt/nvidia/cudnn
mkdir lib
pushd lib
for file in /lib/x86_64-linux-gnu/*cudnn*.so; do
  ln -s $file .
done
popd
mkdir include
pushd include
for file in /usr/include/*cudnn*.h; do
  ln -s $file .
done
popd
popd

# symlink nccl into a private directory
# to avoid include conflicts
mkdir -p /opt/nvidia/nccl
pushd /opt/nvidia/nccl
mkdir lib
pushd lib
for file in /lib/x86_64-linux-gnu/*nccl*.so; do
  ln -s $file .
done
popd
mkdir include
pushd include
for file in /usr/include/*nccl*.h; do
  ln -s $file .
done
popd
popd

ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
