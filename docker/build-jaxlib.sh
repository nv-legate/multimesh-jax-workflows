#! /usr/bin/env bash

set -e

pushd /opt/workspace/jax

BUILD_DIR=$1
BAZEL_CACHE=$2

export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES=sm_80,sm_90a
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_CUDA=1
export TF_NEED_CUTENSOR=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDNN_PATHS=/usr/lib/$(uname -p)-linux-gnu
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-5 | head -c 6)
export TF_CUBLAS_VERSION=$(ls /usr/local/cuda/lib64/libcublas.so.*.*.* | cut -d . -f 3)
export TF_CUDA_MAJOR_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3)
export TF_CUDNN_VERSION=$(echo "${NV_CUDNN_VERSION}" | cut -d . -f 1-3)
export TF_CUDNN_MAJOR_VERSION=$(echo "${NV_CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)

cat > .bazelrc.user << EOF
build:cuda --repo_env=LOCAL_CUDA_PATH="/usr/local/cuda"
build:cuda --repo_env=LOCAL_CUDNN_PATH="/opt/nvidia/cudnn"
build:cuda --repo_env=LOCAL_NCCL_PATH="/opt/nvidia/nccl"
EOF

if [ -z "$BAZEL_CACHE" ]; then
    remote_cache_option=""
else
    remote_cache_option="--bazel_options=--remote_cache=${BAZEL_CACHE}"
fi

conda run --no-capture-out -n legere python build/build.py build \
  --bazel_startup_options=--batch \
  --bazel_options=--override_repository=xla=/opt/workspace/xla \
  $remote_cache_option \
  --bazel_startup_options=--output_base=/opt/build/jaxlib \
  --output_path=/opt/build/jaxlib \
  --use_clang \
  --clang_path=/usr/lib/llvm-17/bin/clang \
  --wheels=jaxlib,jax-cuda-plugin \
  --cuda_version=$TF_CUDA_VERSION \
  --cudnn_version=$TF_CUDNN_VERSION \
  --cuda_compute_capabilities=$TF_CUDA_COMPUTE_CAPABILITIES
