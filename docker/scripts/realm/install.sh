#! /usr/bin/env bash

set -e

pushd /opt/workspace/realm

conda run --no-capture-out -n legere cmake --install /opt/build/realm

popd
