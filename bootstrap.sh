#! /usr/bin/env bash

python -m pip install -r requirements.txt

git submodule update --init --recursive

pushd docker
./apply-patches.sh
popd

