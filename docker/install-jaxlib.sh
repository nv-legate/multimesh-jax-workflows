#! /usr/bin/env bash

BUILD_DIR=$1

pushd $BUILD_DIR
conda run --no-capture-out -n legere python -m pip install *.whl --force-reinstall --no-deps
popd
