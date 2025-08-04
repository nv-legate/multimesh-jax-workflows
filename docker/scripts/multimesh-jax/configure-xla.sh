#! /usr/bin/env bash

pushd /opt/workspace/xla

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

conda run --no-capture-out -n legere python configure.py \
  --backend CUDA \
  --host_compiler CLANG \
  --clang_path `which clang-17` \
  --nccl \
  --cuda_version $TF_CUDA_VERSION \
  --cudnn_version $TF_CUDNN_VERSION \
  --local_cuda_path /usr/local/cuda \
  --local_cudnn_path /opt/nvidia/cudnn \
  --local_nccl_path /opt/nvidia/nccl \
  --cuda_compute_capabilities=sm_80,sm_90a

python_version=$(/opt/install/miniconda/envs/legere/bin/python --version | awk '{print $2}' | cut -d . -f 1-2)
echo "build --repo_env HERMETIC_PYTHON_VERSION=${python_version}" >> xla_configure.bazelrc
echo "build:cuda --@local_config_cuda//cuda:include_cuda_libs=false" >> xla_configure.bazelrc
echo "build:cuda_libraries_from_stubs --@local_config_cuda//cuda:include_cuda_libs=false" >> xla_configure.bazelrc

cat <<EOF >> xla_configure.bazelrc
build:asan --strip=never
build:asan --copt -fsanitize=address
build:asan --copt -DADDRESS_SANITIZER
build:asan --copt -O2
build:asan --copt -g
build:asan --copt -fno-omit-frame-pointer
build:asan --linkopt -fsanitize=address
EOF

popd

# stash the configure for later
cp /opt/workspace/xla/xla_configure.bazelrc /opt
