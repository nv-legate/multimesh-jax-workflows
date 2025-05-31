ARG CUDA_VERSION="12.8.1"
ARG CUDNN_VERSION=""
ARG FINAL_IMAGE_BASE=distribute
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu22.04 AS base

SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

ENV CUDA_PATH="/usr/local/cuda/lib64/stubs"
RUN --mount=type=bind,source=scripts/apt,target=/opt/apt \
  /opt/apt/install.sh

FROM base AS nccl_build
COPY nccl /opt/nccl
COPY scripts/nccl/install-build-deps.sh /opt/nccl
COPY scripts/nccl/build.sh /opt/nccl
COPY scripts/nccl/install.sh /opt/nccl
RUN /opt/nccl/install-build-deps.sh
RUN cd /opt/nccl && /opt/nccl/build.sh

FROM base AS nccl_install
RUN --mount=type=cache,from=nccl_build,source=/opt/nccl,target=/opt/nccl \
  cd /opt/nccl && /opt/nccl/install.sh

FROM base AS mofed_download
RUN mkdir -p /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64
COPY scripts/mofed/install.sh /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64
COPY scripts/mofed/install-build-deps.sh /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64
RUN /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64/install-build-deps.sh
RUN cd /opt && \
  curl -fsSL http://content.mellanox.com/ofed/MLNX_OFED-5.4-3.5.8.0/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64.tgz | tar -xz

FROM nccl_install AS mofed_install
COPY ibdev2netdev /usr/bin
RUN --mount=type=cache,from=mofed_download,source=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64,target=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64 \
  /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64/install.sh

FROM mofed_install AS ucx_build
COPY ucx /opt/ucx
COPY scripts/ucx/install-build-deps.sh /opt/ucx
COPY scripts/ucx/build.sh /opt/ucx
RUN /opt/ucx/install-build-deps.sh
RUN /opt/ucx/build.sh

FROM mofed_install AS ucx_install
RUN --mount=type=cache,from=ucx_build,source=/opt/ucx,target=/opt/ucx \
  cd /opt/ucx && make -j install

FROM ucx_install AS ucc_build
COPY ucc /opt/ucc
COPY scripts/ucc/install-build-deps.sh /opt/ucc
COPY scripts/ucc/build.sh /opt/ucc
RUN /opt/ucc/install-build-deps.sh
RUN /opt/ucc/build.sh

FROM ucx_install AS ucc_install
RUN --mount=type=cache,from=ucc_build,source=/opt/ucc,target=/opt/ucc \
  cd /opt/ucc && make -j install

FROM ucc_install AS symlink_cuda
COPY scripts/symlink-cudalibs.sh /opt
RUN /opt/symlink-cudalibs.sh

FROM base AS conda_download
RUN mkdir -p /opt/conda
COPY scripts/conda/install-build-deps.sh /opt/conda
COPY scripts/conda/download.sh /opt/conda
COPY scripts/conda/install.sh /opt/conda
RUN /opt/conda/install-build-deps.sh
RUN /opt/conda/download.sh

FROM symlink_cuda AS conda_base
RUN --mount=type=cache,from=conda_download,source=/opt/conda,target=/opt/conda \
  /opt/conda/install.sh

ENV LD_LIBRARY_PATH="/opt/install/miniconda/envs/legere/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/install/miniconda/bin:${PATH}"

FROM conda_base AS te_build
COPY te /opt/te
COPY scripts/te/build.sh /opt/te
COPY scripts/te/install.sh /opt/te
RUN --mount=type=bind,source=te-xla,target=/opt/xla \
  /opt/te/build.sh

# Mount a clean XLA source tree distinct from the XLA
# tree for the plugin. TE only uses the XLA code
# to query for certain parameters. Any changes to
# XLA plugin code would otherwise trigger a complete
# rebuild of TransformerEngine, which is very expensive
FROM conda_base AS te_install
RUN --mount=type=bind,source=te-xla,target=/opt/xla \
  --mount=type=cache,from=te_build,source=/opt/te,target=/opt/te \
  /opt/te/install.sh

FROM te_install AS maxtext_install
COPY workspace/maxtext /opt/maxtext
COPY scripts/maxtext/install.sh /opt/maxtext
RUN /opt/maxtext/install.sh

FROM maxtext_install AS build_custom_jaxlib
ARG BAZEL_CACHE=grpc://host.docker.internal:9092
COPY workspace/xla /opt/workspace/xla
COPY workspace/jax /opt/workspace/jax
COPY scripts/jaxlib/install-build-deps.sh /opt/workspace/jax
COPY scripts/jaxlib/build.sh /opt/workspace/jax
COPY scripts/jaxlib/install.sh /opt/workspace/jax
RUN /opt/workspace/jax/install-build-deps.sh
RUN /opt/workspace/jax//build.sh ${BAZEL_CACHE}

FROM maxtext_install AS install_custom_jaxlib
RUN conda run -n legere python -m pip uninstall -y jaxlib
RUN --mount=type=cache,from=build_custom_jaxlib,source=/opt/build/jaxlib,target=/opt/build/jaxlib \
  --mount=type=cache,from=build_custom_jaxlib,source=/opt/workspace/jax,target=/opt/workspace/jax \
  /opt/workspace/jax/install.sh

FROM install_custom_jaxlib AS reinstall_custom_jax
RUN conda run -n legere python -m pip uninstall -y jax
COPY workspace/jax /opt/workspace/jax
RUN cd /opt/workspace/jax && \
  conda run --no-capture-out -n legere python -m pip install -e . --force-reinstall --no-deps

FROM reinstall_custom_jax AS devcontainer_base
COPY cpm /opt/cpm
COPY rapids-cmake /opt/rapids-cmake

FROM devcontainer_base AS devcontainer_test
COPY workspace/xla /opt/workspace/xla
COPY workspace/realm /opt/workspace/realm
COPY workspace/zuku /opt/workspace/zuku
COPY workspace/multimesh-jax /opt/workspace/multimesh-jax

FROM reinstall_custom_jax AS build_realm
ARG BUILD_TYPE=Release
COPY workspace/realm /opt/workspace/realm
COPY scripts/realm/* /opt/workspace/realm
RUN /opt/workspace/realm/install-build-deps.sh
RUN --mount=type=cache,target=/ccache/realm \
  /opt/workspace/realm/configure.sh ${BUILD_TYPE}
RUN --mount=type=cache,target=/ccache/realm \
  /opt/workspace/realm/build.sh

FROM reinstall_custom_jax AS install_realm
RUN --mount=type=cache,from=build_realm,source=/opt/build/realm,target=/opt/build/realm \
  --mount=type=cache,from=build_realm,source=/opt/workspace/realm,target=/opt/workspace/realm \
  --mount=type=cache,from=build_realm,source=/opt/lib,target=/opt/lib \
  --mount=type=cache,target=/ccache/realm \
  /opt/workspace/realm/install.sh

FROM install_realm AS build_zuku
ARG BUILD_TYPE=Release
COPY workspace/zuku /opt/workspace/zuku
COPY scripts/zuku/* /opt/workspace/zuku
COPY cpm /opt/cpm
COPY rapids-cmake /opt/rapids-cmake
RUN /opt/workspace/zuku/install-build-deps.sh
RUN --mount=type=cache,target=/ccache/zuku \
  /opt/workspace/zuku/configure.sh ${BUILD_TYPE}
RUN --mount=type=cache,target=/ccache/zuku \
  /opt/workspace/zuku/build.sh

FROM install_realm AS install_zuku
RUN --mount=type=cache,target=/ccache/zuku \
    --mount=type=cache,from=build_zuku,source=/opt/build/zuku,target=/opt/build/zuku \
    --mount=type=cache,from=build_zuku,source=/opt/lib,target=/opt/lib \
    --mount=type=cache,from=build_zuku,source=/opt/workspace/zuku,target=/opt/workspace/zuku \
  /opt/workspace/zuku/install.sh

FROM install_zuku AS build_multimesh_plugin
ARG BUILD_TYPE=Release
ARG BAZEL_CACHE=grpc://host.docker.internal:9092
COPY workspace/multimesh-jax /opt/workspace/multimesh-jax
COPY scripts/multimesh-jax/* /opt/workspace/multimesh-jax
COPY cpm /opt/cpm
COPY rapids-cmake /opt/rapids-cmake
COPY workspace/xla /opt/workspace/xla
RUN /opt/workspace/multimesh-jax/install-build-deps.sh
RUN /opt/workspace/multimesh-jax/configure-xla.sh
RUN /opt/workspace/multimesh-jax/configure.sh ${BUILD_TYPE} ${BAZEL_CACHE}
RUN \
  --mount=type=cache,from=build_zuku,source=/opt/workspace/zuku,target=/opt/workspace/zuku \
  --mount=type=cache,from=build_realm,source=/opt/workspace/realm,target=/opt/workspace/realm \
  /opt/workspace/multimesh-jax/build.sh

FROM install_zuku AS install_multimesh_plugin
COPY rapids-cmake /opt/rapids-cmake
RUN --mount=type=cache,from=build_multimesh_plugin,source=/opt/build/multimesh-jax,target=/opt/build/multimesh-jax \
  --mount=type=cache,from=build_multimesh_plugin,source=/opt/workspace/xla,target=/opt/workspace/xla \
  --mount=type=cache,from=build_multimesh_plugin,source=/opt/workspace/multimesh-jax,target=/opt/workspace/multimesh-jax \
  --mount=type=cache,from=build_multimesh_plugin,source=/opt/lib,target=/opt/lib \
  --mount=type=cache,from=build_multimesh_plugin,source=/root/.cache/bazel,target=/root/.cache/bazel \
    /opt/workspace/multimesh-jax/install.sh

FROM install_multimesh_plugin AS install_tools
RUN --mount=type=bind,source=scripts/tools,target=/opt/scripts/tools \
  /opt/scripts/tools/install.sh jupyter

FROM install_tools AS distribute
COPY apt-base /opt/source/apt-base
RUN --mount=type=bind,source=scripts,target=/opt/scripts \
  mkdir -p /opt/source && cd /opt/source && /opt/scripts/download-all-sources
COPY nccl /opt/source/nccl
COPY ucc /opt/source/ucc
COPY ucx /opt/source/ucx
COPY te /opt/source/te
COPY workspace/xla /opt/source/xla

FROM ${FINAL_IMAGE_BASE} as final_image
COPY workspace/maxtext-scripts/run.py /opt/maxtext/run.py

COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "/bin/bash" ]
