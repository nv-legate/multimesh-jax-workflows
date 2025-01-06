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

Scripts for running `PaxML` and `MaxText` are included to simplify the process
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

