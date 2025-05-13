#! /usr/bin/env bash

editable_flag=${1:-}

export NVTE_FRAMEWORK=jax

pushd /opt/te
conda run -n legere python -m pip install dist/*.whl --no-deps --no-cache-dir
