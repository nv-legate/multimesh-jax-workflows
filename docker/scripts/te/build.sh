#! /usr/bin/env bash

editable_flag=${1:-}

export NVTE_FRAMEWORK=jax

pushd /opt/te

conda run -n legere python -m pip install cmake scikit-build --no-cache-dir

conda run -n legere --no-capture-output \
  python setup.py bdist_wheel
