#! /usr/bin/env bash

git submodule set-branch --branch legate-main docker/xla
git submodule set-branch --branch main docker/legate-jax
git submodule set-branch --branch main docker/zuku

git submodule update --init --recursive

pushd docker
./apply-patches.sh
popd

topdir=`pwd`
# convert the submodules into full directories
for folder in "xla" "legate-jax" "jax" "te" "realm" "zuku"; do
  pushd docker/$folder
  git fetch --unshallow
  gitstub=.git
  fullgitdir=$topdir/.git/modules/docker/$folder
  rm $gitstub
  mkdir -p $gitstub
  cp -r $fullgitdir/* $gitstub
  grep -v worktree $gitstub/config > new-config
  mv new-config $gitstub/config
  popd
done
