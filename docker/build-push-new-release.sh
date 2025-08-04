#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

./build.py \
  --stage final_image \
  --tag $1 \
  --repo ghcr.io/nv-legate \
  --name multimesh-jax \
  --include-source

# before pushing, do some basic validation of the image
docker run \
  -p 8675:8675 \
  -w /opt/workspace/multimesh-jax/docs/notebooks \
  ghcr.io/nv-legate/multimesh-jax:$1 \
  jupyter nbconvert \
    --to notebook --execute mpmd-multimesh.ipynb

docker push ghcr.io/nv-legate/multimesh-jax:$1
