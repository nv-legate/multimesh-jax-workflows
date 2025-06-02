#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


image=${1:-ghcr.io/nv-legate/multimesh-jax:v0.1.1}
WORKDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run \
  --entrypoint /opt/entrypoint.sh \
  --mount type=bind,source=$WORKDIR,target=/workspace \
  -w /workspace \
  $image \
  ./validate-cpu.sh
