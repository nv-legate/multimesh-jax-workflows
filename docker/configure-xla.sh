#! /usr/bin/env bash

export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_NCCL_VERSION=$(ls /lib/x86_64-linux-gnu/libnccl.so.*.*.* | cut -d . -f 3-5)
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-4)
TF_CUDNN_MAJOR_VERSION=$(grep "#define CUDNN_MAJOR" /usr/include/cudnn_version.h | awk '{print $3}')
TF_CUDNN_MINOR_VERSION=$(grep "#define CUDNN_MINOR" /usr/include/cudnn_version.h | awk '{print $3}')
TF_CUDNN_PATCHLEVEL_VERSION=$(grep "#define CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h | awk '{print $3}')
export TF_CUDNN_VERSION="${TF_CUDNN_MAJOR_VERSION}.${TF_CUDNN_MINOR_VERSION}.${TF_CUDNN_PATCHLEVEL_VERSION}"
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export CC_OPT_FLAGS=--Wno-sign-compare
export TF_SET_ANDROID_WORKSPACE=0

# symlink cudnn into a private directory
# to avoid include conflicts
mkdir -p /opt/nvidia/cudnn
pushd /opt/nvidia/cudnn
mkdir lib
pushd lib
for file in /lib/x86_64-linux-gnu/*cudnn*.so; do
  ln -s $file .
done
popd
mkdir include
pushd include
for file in /usr/include/*cudnn*.h; do
  ln -s $file .
done
popd
popd

# symlink nccl into a private directory
# to avoid include conflicts
mkdir -p /opt/nvidia/nccl
pushd /opt/nvidia/nccl
mkdir lib
pushd lib
for file in /lib/x86_64-linux-gnu/*nccl*.so; do
  ln -s $file .
done
mkdir include
pushd include
for file in /usr/include/*nccl*.h; do
  ln -s $file .
done
popd
popd

conda run --no-capture-out -n legere python configure.py \
  --backend CUDA \
  --host_compiler CLANG \
  --clang_path `which clang-15` \
  --nccl \
  --local_cuda_path /usr/local/cuda \
  --local_cudnn_path /opt/nvidia/cudnn \
  --local_nccl_path /opt/nvidia/nccl \
  --nccl_version=$TF_NCCL_VERSION \
  --cuda_compute_capabilities=sm_80,sm_90a

python_version=$(/opt/install/miniconda/envs/legere/bin/python --version | awk '{print $2}' | cut -d . -f 1-2)
echo "build --repo_env HERMETIC_PYTHON_VERSION=${python_version}" >> xla_configure.bazelrc
