# MultiMesh for Jax Workflows

MultiMesh for JAX provides a framework for creating `task` contexts within jitted computations,
allowing different subcomputations to be placed on different GPU submeshes. These
`task` computations can be combined inside a global `jit` with data resharding across submeshes
occurring automatically. MultiMesh therefore enables pipeline parallelism to be easily expressed.
This repository provides a monorepo and associated workflows for creating a [MaxText](https://github.com/AI-Hypercomputer/maxtext)
stack running with [MultiMesh](https://github.com/nv-legate/multimesh-jax).

## Prebuilt Containers

Prebuilt containers are published to the [MultiMesh Github container registry](https://github.com/nv-legate/multimesh-jax/pkgs/container/multimesh-jax).
The most recent release can be pulled:

```bash
$ docker pull ghcr.io/nv-legate/multimesh-jax:v0.1.1
```

## Docker Builds

Instructions for building containers can be found [here](docker/README.md).
To build the container, numerous submodules need to be downloaded and, in some cases, patched.
To do so, run the `./bootstrap.sh` script in the top folder.
We recommend using the [build driver script](docker/build.py).
For a full list of options, one can run `build.py --help`.

The most common build workflow will be:
```bash
multimesh-jax-workflows$ ./bootstrap.sh
multimesh-jax-workflows$ cd docker
multimesh-jax-workflows/docker$ ./build.py --tag <TAG> --upload --repo <REPO>
```

which builds an image named `<REPO>:<TAG>` and uploads it,
assuming that `<REPO>` points to a valid container registry.

## MaxText Configs

The main framework integrated with MultiMesh for Jax is [MaxText](https://github.com/AI-Hypercomputer/maxtext).
A default Docker build will produce an image with a MaxText
installation. Please see [the README](maxtext/README.md) for
instructions on running MaxText with the provided helper scripts.


## Open as Devcontainer

The top-level multimesh-jax.code-workspace can be opened in vscode,
which should prompt to `Reopen in container`. Select this option.
This will load vscode in a new devcontainer with all base dependencies installed.
Startup scripts will then configure all builds and execute an initial build
of the environment. The startup scripts point Bazel and CMake to build
caches on your local system.

* The initial base container download may take a long time on the first download
* The startup scripts may take a few minutes to up to an hour depending on how
  much of the build is available in the build cache.

Once the workspace is open, it now contains a complete vscode development environment.

### Remote cache

It is highly recommended to set up a remote bazel cache that is shared across
containers. There may be an initial warmup in the first devcontainer, but
subsequent builds should be quick.  It should be sufficient to run:

```bash
multimesh-jax-workflows/docker $ ./start-cache.sh
```

## Driver script for MaxText

A script for running [MaxText](docker/workspace/maxtext-scripts/run.py) is included to simplify the process
of tuning parameters. A full list of options can be required by running:

```bash
multimesh-jax-workflows $ maxtext/run.py --help
```

There are several options for configuring the parallelism:

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

## Running smoke tests locally for MaxText

To run an example job in the container locally, example scripts are included in the repo.

### MaxText with 2 GPUs

A [script](docker/workspace/maxtext-scripts/validate-gpu.sh) for running a small job with TP=2 is included.
The container can be launched from the top-level directory as:

```bash
docker run \
  --entrypoint /opt/entrypoint.sh \
  --mount type=bind,source=$(pwd)/docker/workspace/maxtext-scripts,target=/workspace \
  -w /workspace \
  --gpus 2 \
  ghcr.io/nv-legate/multimesh-jax:v0.1.1 \
  ./validate-gpu.sh
```

### MaxText with 8 CPUs

A [script](docker/workspace/maxtext-scripts/validate-cpu.sh) for running a small job with DP=2, PP=2, TP=2 is included.
Currently the container requires CUDA present even if running a
CPU-only job. To launch the job:

```bash
docker run \
  --entrypoint /opt/entrypoint.sh \
  --mount type=bind,source=$(pwd)/docker/workspace/maxtext-scripts,target=/workspace \
  -w /workspace \
  ghcr.io/nv-legate/multimesh-jax:v0.1.1 \
  ./validate-cpu.sh
```

### Slurm scripts

Scripts are included that are intended to be used inside a Slurm job, e.g.

```
srun -N <N> ... run-dgx-h100.sh
```

Scripts are included for:

* [Llama3 70B on 64 GPUs](docker/workspace/maxtext-scripts/llama3-70b-64gpus/run-dgx-h100.sh)
* [Llama3 8B on 16 GPUs](docker/workspace/maxtext-scripts/llama3-8b-16gpus/run-dgx-h100.sh)
* [GPT3 175B on 64 GPUs](docker/workspace/maxtext-scripts/gpt3-175b-64gpus/run-dgx-h100.sh)

Each folder contains a template Slurm batch script for launching containers.
