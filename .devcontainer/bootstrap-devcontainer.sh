#! /usr/bin/env bash

mkdir -p ~/.cache/ccache/zuku
mkdir -p ~/.cache/ccache/realm
mkdir -p ~/.cache/ccache/multimesh-jax

topdir=$1
# convert the submodules into full directories
pushd $topdir
for folder in "xla" "multimesh-jax" "jax" "realm" "zuku"; do
  if [ ! -e "docker/workspace/$folder/.git" ]; then
    git submodule update --init --recursive -- docker/workspace/$folder
  fi
done
popd

pushd $topdir/docker
./apply-patches.sh
popd
