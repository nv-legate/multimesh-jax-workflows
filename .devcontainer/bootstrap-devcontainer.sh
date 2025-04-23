#! /usr/bin/env bash

mkdir -p ~/.cache/ccache/zuku
mkdir -p ~/.cache/ccache/realm
mkdir -p ~/.cache/ccache/legate-jax

topdir=$1
# convert the submodules into full directories
for folder in "xla" "legate-jax" "jax" "realm" "zuku"; do
  pushd $topdir
  git submodule update --init --recursive -- docker/workspace/$folder
  popd
done

pushd $topdir/docker
./apply-patches.sh
popd
