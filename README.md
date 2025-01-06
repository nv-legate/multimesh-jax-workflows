# Jax Workflows

## Getting started

To build the container, numerous submodules need to be downloaded and, in some cases, patched.
To do so, run the `./bootstrap.sh` script in the top folder.

## Docker Builds

Instructions for building containers can be found [here](docker/README.md).
We recommend using the [build driver script](docker/build.py).
For a full list of options, one can run `build.py --help`.

## Jax APIs documentation and tutorials

Documentation on the Jax APIs can be found [here](http://sw-mobile-docs/cllr/legate-jax/).

## Driver scripts for running PaxML/MaxText

Scripts for running [PaxML](paxml/run.py) and [MaxText](maxtext/run.py) are included to simplify the process
of tuning parameters. A full list of options can be required by running:

```
legate-jax-workflows $ paxml/run.py --help
legate-jax-workflows $ maxtext/run.py --help
```

Most of the options for configuring Legate-Jax and the different parallelism
will be the same:

* `--dp <N>`: The degree of data parallelism
* `--tp <N>`: The degree of tensor parallelism
* `--fsdp <N>`: The amount of fully-sharded data parallelism
* `--pp <N>`: The amount of pipeline-parallelism

Currently, it is recommended to run in GPU/process mode. In this case,
parameters should be:

* `--nodes <N>`: The total number of processes (GPUs with proc/GPU). The product of DP x FSDP x PP x TP
  should match this number.
* `--gpus 1`: A single GPU per process.
* `--cpus 1`: A companion host for each GPU

For configuring pipeline parallelism, the most important parameters are:

 * `--batch-size <N>`: The global batch size. Most often, this will be 2x or 4x the total no. of GPUs
 * `--microbatch-size <N>`: The microbatch size per tensor-parallel domain. Each domain will compute on a batch of shape `(Microbatch Size, Sequence Length)`.
    For a batch size of 1024, data-parallelism 8, and microbatch size 4 there will be a total of 32 microbatches.

PaxML and MaxText will each have unique parameters for selecting models and configuring
features of the model, such as checkpoint/recomputation strategy.

## Running smoke tests locally for PaxML and MaxText

To run an example job in the container locally, example scripts are included in the repo.

### MaxText with 2 GPUs

A [script](maxtext/validate-gpu.sh) for running a small job with TP=2 is included.
The container can be launched from the top-level directory as:

```
docker run \
  --cap-add SYS_ADMIN \
  --entrypoint /opt/entrypoint.sh \
  --net=host \
  --add-host=host.docker.internal:host-gateway \
  --mount type=bind,source="$(pwd)"/maxtext,target=/workspace \
  -w /workspace \
  --gpus 2 \
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev:maxtext \
  ./validate-gpu.sh
```

### MaxText with 8 CPUs

A [script](maxtext/validate-cpu.sh) for running a small job with DP=2, PP=2, TP=2 is included.
Currently the container requires CUDA present even if running a
CPU-only job. To launch the job:

```
docker run \
  --cap-add SYS_ADMIN \
  --entrypoint /opt/entrypoint.sh \
  --net=host \
  --add-host=host.docker.internal:host-gateway \
  --mount type=bind,source="$(pwd)"/maxtext,target=/workspace \
  -w /workspace \
  --gpus 2 \
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev:maxtext \
  ./validate-cpu.sh
```

### PaxML with 2 GPUs

A [script](paxml/validate-gpu.sh) for running a small job with TP=2 is included.
PaxML is currently most stable with GPU/process:

```
docker run \
  --cap-add SYS_ADMIN \
  --entrypoint /opt/entrypoint.sh \
  --net=host \
  --add-host=host.docker.internal:host-gateway \
  --mount type=bind,source="$(pwd)"/paxml,target=/workspace \
  --gpus 2 \
  -w /workspace \
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev:paxml \
  mpirun -n 2 --allow-run-as-root validate-gpu.sh
```

### PaxML with 8 CPUs

A [script](paxml/validate-cpu.sh) for running a small job with PP=2, TP=4 is included.
Currently the container requires CUDA present even if running a
CPU-only job. To launch the job:

```
docker run \
  --cap-add SYS_ADMIN \
  --entrypoint /opt/entrypoint.sh \
  --net=host \
  --add-host=host.docker.internal:host-gateway \
  --mount type=bind,source="$(pwd)"/paxml,target=/workspace \
  --gpus 2 \
  -w /workspace \
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev:paxml \
  ./validate-cpu.sh
```

## Large runs on a DGX

Scripts and config files are included for running larger jobs
with hwloc bindings for a DGX H100.

* [PaxML GPT3-175B for 64 GPUs](paxml/gpt3-175b-64gpus)
* [MaxText GPT3-175B for 64 GPUs](maxtext/gpt3-175b-64gpus)
