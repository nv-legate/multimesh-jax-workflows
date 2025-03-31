ARG CUDA_VERSION="12.8.1"
ARG CUDNN_VERSION=""
ARG BASE_NAME="maxtext_base"
ARG JAX_PLUGIN_BASE="editable_install_zuku"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu22.04 AS base

SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

COPY install-apt.sh /opt
RUN /opt/install-apt.sh

ENV CUDA_PATH="/usr/local/cuda/lib64/stubs"

FROM base AS nccl_build
COPY nccl /opt/nccl
COPY build-nccl.sh /opt/nccl
COPY install-nccl.sh /opt/nccl
RUN cd /opt/nccl && ./build-nccl.sh

FROM base AS nccl_install
RUN --mount=type=cache,from=nccl_build,source=/opt/nccl,target=/opt/nccl \
  cd /opt/nccl && ./install-nccl.sh

FROM nccl_install AS bazel_install
COPY install-bazel.sh /opt
RUN /opt/install-bazel.sh

FROM bazel_install AS conda_base
COPY install-conda.sh /opt
RUN /opt/install-conda.sh

ENV LD_LIBRARY_PATH="/opt/install/miniconda/envs/legere/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/install/miniconda/bin:${PATH}"

FROM conda_base AS mellanox_download

RUN cd /opt && \
  curl -fsSL http://content.mellanox.com/ofed/MLNX_OFED-5.4-3.5.8.0/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64.tgz | tar -xz

FROM conda_base AS mellanox_install
COPY ibdev2netdev /usr/bin
COPY install-mofed.sh /opt
RUN --mount=type=cache,from=mellanox_download,source=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64,target=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64 \
  /opt/install-mofed.sh

FROM mellanox_install AS ucx_build
COPY ucx /opt/ucx
COPY build-ucx.sh /opt/ucx
RUN /opt/ucx/build-ucx.sh

FROM mellanox_install AS ucx_install
RUN --mount=type=cache,from=ucx_build,source=/opt/ucx,target=/opt/ucx \
  cd /opt/ucx && make -j install

FROM ucx_install AS ucc_build
COPY ucc /opt/ucc
COPY build-ucc.sh /opt/ucc
RUN /opt/ucc/build-ucc.sh

FROM ucx_install AS ucc_install
RUN --mount=type=cache,from=ucc_build,source=/opt/ucc,target=/opt/ucc \
  cd /opt/ucc && make -j install

FROM ucc_install AS symlink_cuda
COPY symlink-cudalibs.sh /opt
RUN /opt/symlink-cudalibs.sh

FROM symlink_cuda AS te_install
COPY te /opt/te
COPY install-te.sh /opt
RUN conda run -n legere python -m pip install cmake
# Mount a clean XLA source tree distinct from the XLA
# tree for the plugin. TE only uses the XLA code
# to query for certain parameters. Any changes to
# XLA plugin code would otherwise trigger a complete
# rebuild of TransformerEngine, which is very expensive
RUN --mount=type=bind,source=te-xla,target=/opt/xla \
  /opt/install-te.sh -e

FROM te_install AS install_clang
COPY install-clang.sh /opt
RUN /opt/install-clang.sh
RUN ln -s /usr/bin/clangd-17 /usr/bin/clangd

FROM install_clang AS maxtext_base
COPY maxtext /opt/maxtext
RUN conda run -n legere python -m pip install flit-core
RUN cd /opt/maxtext && conda run -n legere python -m pip install -e . --no-build-isolation --force-reinstall

FROM ${BASE_NAME} AS setup_devcontainer_build
ARG BUILD_BASE_DIR=/opt/build
ARG LIB_DIR=/opt/lib
ARG BAZEL_CACHE=grpc://host.docker.internal:9092
COPY build-jaxlib.sh install-jaxlib.sh /opt

FROM setup_devcontainer_build AS build_custom_jaxlib
COPY workspace/xla /opt/workspace/xla
COPY workspace/jax /opt/workspace/jax
RUN /opt/build-jaxlib.sh ${BUILD_BASE_DIR}/jaxlib ${BAZEL_CACHE}

FROM setup_devcontainer_build AS install_custom_jaxlib
RUN conda run -n legere python -m pip uninstall -y jaxlib
RUN --mount=type=cache,from=build_custom_jaxlib,source=${BUILD_BASE_DIR}/jaxlib,target=${BUILD_BASE_DIR}/jaxlib \
  /opt/install-jaxlib.sh ${BUILD_BASE_DIR}/jaxlib

FROM install_custom_jaxlib AS reinstall_custom_jax
RUN conda run -n legere python -m pip uninstall -y jax
RUN --mount=type=cache,from=build_custom_jaxlib,source=/opt/workspace/jax,target=/opt/workspace/jax \
  cd /opt/workspace/jax && \
  conda run --no-capture-out -n legere python -m pip install . --force-reinstall --no-deps

# rename the stage
FROM reinstall_custom_jax AS devcontainer_base
COPY rapids-cmake /opt/rapids-cmake
COPY cpm /opt/cpm
COPY \
   configure-realm.sh build-realm.sh install-realm.sh \
   configure-zuku.sh build-zuku.sh \
   configure-xla.sh xla-compile-commands.sh \
   configure-legate-jax.sh build-legate-jax.sh install-legate-jax.sh \
   startup-devcontainer.sh \
   /opt

FROM devcontainer_base AS setup_realm
ARG LEGATE_BUILD_TYPE=Release
COPY workspace/realm /opt/workspace/realm
RUN --mount=type=cache,target=/ccache/realm \
  /opt/configure-realm.sh ${LEGATE_BUILD_TYPE} ${BUILD_BASE_DIR}/realm ${LIB_DIR}

FROM setup_realm AS build_realm
RUN --mount=type=cache,target=/ccache/realm \
  /opt/build-realm.sh ${BUILD_BASE_DIR}/realm

FROM setup_realm AS editable_install_realm
RUN --mount=type=cache,from=build_realm,source=${BUILD_BASE_DIR}/realm,target=${BUILD_BASE_DIR}/realm \
  --mount=type=cache,target=/ccache/realm \
  /opt/build-realm.sh ${BUILD_BASE_DIR}/realm

FROM editable_install_realm AS setup_zuku
COPY workspace/zuku /opt/workspace/zuku
RUN --mount=type=cache,target=/ccache/zuku \
    --mount=type=bind,from=build_realm,source=${BUILD_BASE_DIR}/realm,target=${BUILD_BASE_DIR}/realm \
  /opt/configure-zuku.sh ${LEGATE_BUILD_TYPE} ${BUILD_BASE_DIR}/zuku ${BUILD_BASE_DIR}/realm ${LIB_DIR}

FROM setup_zuku AS build_zuku
RUN --mount=type=cache,target=/ccache/zuku \
    --mount=type=bind,from=build_realm,source=${BUILD_BASE_DIR}/realm,target=${BUILD_BASE_DIR}/realm \
  /opt/build-zuku.sh ${BUILD_BASE_DIR}/zuku

FROM setup_zuku AS editable_install_zuku
RUN --mount=type=cache,target=/ccache/zuku \
    --mount=type=bind,from=build_realm,source=${BUILD_BASE_DIR}/realm,target=${BUILD_BASE_DIR}/realm \
    --mount=type=cache,from=build_zuku,source=${BUILD_BASE_DIR}/zuku,target=${BUILD_BASE_DIR}/zuku \
  /opt/build-zuku.sh ${BUILD_BASE_DIR}/zuku

FROM ${JAX_PLUGIN_BASE} AS setup_jax_plugin
COPY workspace/legate-jax /opt/workspace/legate-jax
COPY workspace/xla /opt/workspace/xla
RUN /opt/configure-xla.sh \
    && /opt/configure-legate-jax.sh ${LEGATE_BUILD_TYPE} ${BUILD_BASE_DIR}/legate-jax ${BUILD_BASE_DIR}/realm ${BUILD_BASE_DIR}/zuku ${LIB_DIR} ${BAZEL_CACHE}

FROM setup_jax_plugin AS build_jax_plugin
RUN --mount=type=cache,target=/ccache/legate-jax \
    --mount=type=cache,from=build_realm,source=${BUILD_BASE_DIR}/realm,target=${BUILD_BASE_DIR}/realm \
    --mount=type=cache,from=build_zuku,source=${BUILD_BASE_DIR}/zuku,target=${BUILD_BASE_DIR}/zuku \
    /opt/build-legate-jax.sh ${BUILD_BASE_DIR}/legate-jax

FROM setup_jax_plugin AS editable_install_jax_plugin
RUN --mount=type=cache,from=build_jax_plugin,source=${BUILD_BASE_DIR}/legate-jax,target=${BUILD_BASE_DIR}/legate-jax \
  --mount=type=cache,from=build_jax_plugin,source=/opt/workspace/xla,target=/opt/workspace/xla \
  --mount=type=cache,from=build_jax_plugin,source=/root/.cache/bazel,target=/root/.cache/bazel \
    /opt/build-legate-jax.sh ${BUILD_BASE_DIR}/legate-jax \
    && /opt/install-legate-jax.sh ${BUILD_BASE_DIR}/legate-jax -e .

COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "/bin/bash" ]
