#! /usr/bin/env bash

git submodule update --init --recursive

pushd docker
./apply-patches.sh
popd

