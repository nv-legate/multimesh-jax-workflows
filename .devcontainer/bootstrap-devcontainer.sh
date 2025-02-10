#! /usr/bin/env bash

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
