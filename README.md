# Jax Workflows

## Docker Builds

Instructions for building containers can be found [here](docker/README.md).
To build the container, numerous submodules need to be downloaded and, in some cases, patched.
To do so, run the `./bootstrap.sh` script in the top folder.
We recommend using the [build driver script](docker/build.py).
For a full list of options, one can run `build.py --help`.

## Devcontainer

This repository is the base development project for vscode workflows.
The recommended development workflow is to use prebuilt, independent containers
for each PR. To get started, open the repo folder in vscode. On the first time
starting a devcontainer, do the following:

* Install the Docker extension in your vscode
* Create a conda environment, e.g. `dev-workflows` from the provided environment file on your development machine

```bash
legate-jax-workflows $ conda env create -n dev-workflows -f environment.yml
```

* Install all Python dependencies. It is recommend to use the `eos_workflows` dependency as an editable install
that can be easily updated and debugged.

```bash
legate-jax-workflows $ conda activate dev-workflows
legate-jax-workflows $ git clone ssh://git@gitlab-master.nvidia.com:12051/jwilke/eos-workflows.git
legate-jax-workflows $ cd eos-workflows
eos-workflows        $ python -m pip install -e .
eos-workflows        $ cd ..
legate-jax-workflows $ python -m pip install -r requirements.txt
```

* Create a conda environment from `environment.yml` on all remote machines you want to use for testing
* Create a `~/.config/workflows/config.yml` defining parameters for each remote you want to use for testing

Inside the project, the following steps are then required for each new development environment:

### Clone a new copy of this monorepo repository with a descriptive name

The easiest way is to use the pre-configured task `New Workspace In Folder' (shift+cmd+b on Mac)`.
The user is then prompted to give the image name and a unique name for the container.
The chosen name should describe the feature you are going to develop.
A code window should open with the fresh monorepo checkout.

### Open devcontainer in new window

A prompt should appear asking to reopen in container. Select this option.
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
legate-jax-workflows/docker $ ./start-cache.sh
```

### clangd and compile_comands.json for XLA

The main recommended extension is `clangd`. The container comes with a
prebuilt `compile_commands.json` using the hedronvision compile commands
extractor.

## Development Tasks

Most relevent development tasks can be run as vscode tasks.
Tasks are generally either build/test tasks run *inside* the container
or upload/run tasks run *on* the image/container.

### Tasks inside the container

The main tasks inside the container are build and test tasks:

* Build legate-jax plugin
* Run XLA smoke tests to test XLA changes
* Run JAX unit tests to test complete stack changes

### Tasks outside the container

The main tasks outside the container are build and test tasks:

* Build a new image
* Build and upload a new image to a remote repository
* Full validation workflow
  - Build new a image or commit active container
  - Upload container to repo
  - Save sqsh image on platforms where relevant
  - Launch an `srun` job on a remote and validate the result

## Jax APIs documentation and tutorials

Documentation on the Jax APIs can be found [here](http://sw-mobile-docs/cllr/legate-jax/).

## Driver script for MaxText

A script for running [MaxText](maxtext/run.py) is included to simplify the process
of tuning parameters. A full list of options can be required by running:

```
legate-jax-workflows $ maxtext/run.py --help
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

## Large runs on a DGX

Scripts and config files are included for running larger jobs
with hwloc bindings for a DGX H100.

* [MaxText GPT3-175B for 64 GPUs](maxtext/gpt3-175b-64gpus)
