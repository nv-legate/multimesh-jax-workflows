#! /usr/bin/env bash

function patch {
  folder=$1
  patch=$2
  pushd $folder
  git apply ../$patch
  popd
}

patch te      te.patch
patch maxtext maxtext.patch
