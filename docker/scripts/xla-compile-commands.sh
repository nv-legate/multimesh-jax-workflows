#! /usr/bin/env bash

pushd /opt/workspace/xla

bazel run :refresh_compile_commands

popd

