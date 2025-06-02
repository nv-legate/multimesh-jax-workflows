#! /usr/bin/env bash

set -e

pushd /opt/workspace/realm

conda install -c conda-forge ninja ccache

popd
