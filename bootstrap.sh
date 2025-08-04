#! /usr/bin/env bash

python -m pip install -r requirements.txt

git submodule update --init --recursive

if [[ "${JAX_APPLY_PATCHES:-1}" == "1" ]]; then
  echo "Applying patches..."
  pushd docker
  ./apply-patches.sh
  popd
else
  echo "Not applying patches..."
fi
