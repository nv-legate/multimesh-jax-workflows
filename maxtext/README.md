# Running MaxText with Legate-Jax

This folder contains scripts that illustrate running [MaxText](https://github.com/AI-Hypercomputer/maxtext)
with the Legate-Jax library and plugin.  The `run.py` script
encapsulates the numerous, numerous config options for running
MaxText and tries to provide them as documented flags for the script.
Most critically, it provides `--dp`, `--fsdp`, `--pp`, and `--tp`
flags for configuring parallelism in a simple way.

For a full list of options, the user can run:

```
./run.py --help
```

## Llama3-8b

An example script that can be used with the Slurm [SPANK](https://slurm.schedmd.com/containers.html)
plugin [Pyxis](https://github.com/NVIDIA/pyxis)
illustrates running the Llama3-8b model with a DP=2, PP=2, TP=4 configuration. The script
can be launched with a built Docker image and Slurm over 16 GPUs
(2 nodes) of a DGX H100 system as:

```
srun \
  -N 2 \
  --ntasks-per-node 8 \
  --mpi=pmix \
  --container-image <IMAGE> \
  /opt/entrypoint.sh \
  llama3-8b-16gpus/run-dgx-h100.sh
```

Here `<IMAGE>` should be generated from the `build.py` script.

## Llama3-70b

An example script that can be used with Slurm illustrates running
the Llama3-70b model with a PP=16, TP=4 configuration. The script
can be launched with a built Docker image and Slurm over 64 GPUs
(8 nodes) of a DGX H100 system as:

```
srun \
  -N 8 \
  --ntasks-per-node 8 \
  --mpi=pmix \
  --container-image <IMAGE> \
  /opt/entrypoint.sh \
  llama3-70b-64gpus/run-dgx-h100.sh
```
