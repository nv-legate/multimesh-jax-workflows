#! /usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd ${SCRIPT_DIR}

conda run -n legere conda install -c conda-forge -y git pkg-config

conda run -n legere python -m pip install flit-core

conda run -n legere --no-capture-output \
  python -m pip install -e . --force-reinstall --no-build-isolation --no-cache-dir

conda run -n legere conda remove -y git pkg-config
conda run -n legere python -m pip uninstall -y flit-core
conda run -n legere python -m pip cache purge

popd
