#! /usr/bin/env bash

function patch {
  folder=$1
  patch=$2
  pushd $folder
  git apply $patch
  popd
}

patch te      $(pwd)/patches/te/te.patch
patch maxtext $(pwd)/patches/maxtext/maxtext.patch
patch te/3rdparty/cudnn-frontend $(pwd)/patches/cudnn/cudnn.patch
patch workspace/jax $(pwd)/patches/jax/4c7140f.patch
patch workspace/xla $(pwd)/patches/xla/41c2b0ed.patch
patch workspace/realm $(pwd)/patches/realm/realm.patch
