#! /usr/bin/env bash

set -e

pushd /opt/workspace/zuku

conda install -c conda-forge ninja ccache

popd
