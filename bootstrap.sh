#! /usr/bin/env bash

git submodule update --init --recursive

pushd docker
./apply-patches.sh
popd

cp paxml/run.py docker/paxml
cp maxtext/run.py docker/maxtext
