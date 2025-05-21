#! /usr/bin/env bash

tag=$1

./build.py --name multimesh-jax --repo ghcr.io/nv-legate --tag $tag --upload
