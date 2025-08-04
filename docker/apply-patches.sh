#! /usr/bin/env bash

function patch {
  folder=$1
  patch=$2
  pushd $folder
  git apply $patch
  popd
}

patch te      $(pwd)/patches/te/te.patch
patch workspace/maxtext $(pwd)/patches/maxtext/maxtext.patch
patch te/3rdparty/cudnn-frontend $(pwd)/patches/cudnn/cudnn.patch
patch workspace/jax $(pwd)/patches/jax/jax.patch
patch workspace/xla $(pwd)/patches/xla/xla.patch
patch workspace/realm $(pwd)/patches/realm/realm.patch
