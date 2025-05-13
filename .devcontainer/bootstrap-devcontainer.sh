#! /usr/bin/env bash

mkdir -p ~/.cache/ccache/zuku
mkdir -p ~/.cache/ccache/realm
mkdir -p ~/.cache/ccache/multimesh-jax

topdir=$1
# convert the submodules into full directories
for folder in "xla" "multimesh-jax" "jax" "realm" "zuku"; do
  if [ ! -d "$folder" ]; then
    pushd $topdir
    git submodule update --init --recursive -- docker/workspace/$folder
    popd
  fi
done

pushd $topdir/docker
./apply-patches.sh
popd
