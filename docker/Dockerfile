ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG FRAMEWORK="paxml"
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
FROM ucx_install AS symlink_cuda
COPY symlink-cudalibs.sh /opt
RUN /opt/symlink-cudalibs.sh

FROM symlink_cuda AS te_install
COPY te /opt/te
COPY install-te.sh /opt
RUN conda run -n legere python -m pip install cmake
RUN --mount=type=bind,source=workspace/xla,target=/opt/xla \
  /opt/install-te.sh -e

FROM te_install AS install_clang
COPY install-clang.sh /opt
RUN /opt/install-clang.sh
RUN ln -s /usr/bin/clangd-17 /usr/bin/clangd

FROM install_clang AS copy_common_frameworks

COPY chex /opt/chex
COPY flax /opt/flax
COPY clu /opt/clu
COPY optax opt/optax
COPY seqio /opt/seqio
COPY orbax /opt/orbax
COPY praxis /opt/praxis

FROM copy_common_frameworks AS copy_requirements_maxtext
COPY maxtext-requirements.txt /opt/requirements.txt

FROM copy_common_frameworks AS copy_requirements_paxml
COPY paxml-requirements.txt /opt/requirements.txt

FROM copy_requirements_${FRAMEWORK} AS install_python_libs
COPY install-pip-dependencies.sh /opt
RUN /opt/install-pip-dependencies.sh -e

FROM install_python_libs AS maxtext_install
COPY maxtext /opt/maxtext
RUN cd /opt/maxtext && conda run -n legere python -m pip install -e . --no-deps --no-build-isolation --force-reinstall
COPY maxtext-fixup-pip.sh /opt
RUN /opt/maxtext-fixup-pip.sh

FROM ${FRAMEWORK}_install AS copy_scripts
COPY \
   configure-realm.sh build-realm.sh install-realm.sh \
   configure-zuku.sh build-zuku.sh \
   configure-xla.sh xla-compile-commands.sh \
   build-jaxlib.sh install-jaxlib.sh \
   configure-legate-jax.sh build-legate-jax.sh install-legate-jax.sh \
   /opt

FROM copy_scripts AS devcontainer_base
ARG BUILD_BASE_DIR=/opt/build
ARG LIB_DIR=/opt/lib
COPY startup-devcontainer.sh /opt

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

FROM editable_install_zuku AS setup_xla
ENV PATH=$PATH:/opt/bazel
COPY workspace/xla /opt/workspace/xla
RUN /opt/configure-xla.sh

FROM editable_install_zuku AS setup_jax_plugin
COPY workspace/legate-jax /opt/workspace/legate-jax
COPY workspace/xla /opt/workspace/xla
RUN /opt/configure-xla.sh \
    && /opt/configure-legate-jax.sh ${LEGATE_BUILD_TYPE} ${BUILD_BASE_DIR}/legate-jax ${BUILD_BASE_DIR}/realm ${BUILD_BASE_DIR}/zuku ${LIB_DIR}

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

FROM editable_install_jax_plugin AS build_custom_jaxlib
COPY workspace/xla /opt/workspace/xla
COPY workspace/jax /opt/workspace/jax
ARG BAZEL_CACHE=grpc://host.docker.internal:9092
RUN /opt/build-jaxlib.sh ${BAZEL_CACHE} ${BUILD_BASE_DIR}/jaxlib

FROM editable_install_jax_plugin AS install_custom_jaxlib
RUN conda run -n legere python -m pip uninstall -y jaxlib
RUN --mount=type=cache,from=build_custom_jaxlib,source=${BUILD_BASE_DIR}/jaxlib,target=${BUILD_BASE_DIR}/jaxlib \
  /opt/install-jaxlib.sh ${BUILD_BASE_DIR}/jaxlib

FROM install_custom_jaxlib AS reinstall_custom_jax
RUN conda run -n legere python -m pip uninstall -y jax
RUN cd /opt/workspace/jax && \
  conda run --no-capture-out -n legere python -m pip install . --force-reinstall --no-deps

COPY entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "/bin/bash" ]
