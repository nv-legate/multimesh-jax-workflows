#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

./build.py --stage distribute --tag $1 --repo ghcr.io/nv-legate --upload --name multimesh-jax
