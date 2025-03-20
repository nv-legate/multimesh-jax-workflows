#! /usr/bin/env bash

function patch {
  folder=$1
  patch=$2
  pushd $folder
  git apply $patch
  popd
}

patch te      $(pwd)/te.patch
patch maxtext $(pwd)/maxtext.patch
patch te/3rdparty/cudnn-frontend $(pwd)/cudnn.patch
