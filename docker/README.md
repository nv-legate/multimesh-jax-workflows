# Docker Image Instructions

The Dockerfile and scripts in this folder create a working image of [MaxText](https://github.com/AI-Hypercomputer/maxtext).
There are a few basic steps to create a working image:

## Download and patch all relevant repos
Numerous libraries with dependencies on specific commits are required. These are
managed as Git submodules. To run the submodule updates and apply patch files,
run the `bootstrap.sh` in the top-level directory:

```bash
legate-jax-workflows $ ./bootstrap.sh
```

Some of the repos are private to Nvidia and expect an SSH key to have been configured.


## (Optional) Start a remote Bazel cache

The XLA build compiles 1000s of files and can take a very long time. If you need to consistetly
rebuild an image with updates to any of the repos, it is highly recommended to set up a remote bazel cache.
It should be sufficient to run:

```bash
legate-jax-workflows/docker $ ./start-cache.sh
```

This downloads and starts a Docker container with the remote cache running on gRPC port 9092.
The script sets up the cache directory to be `$(pwd)/cache` and sets an upper limit of 30GB.
These parameters can be changed in the script.  The XLA builds generate as much as 20GB of 
files for a clean build.  

The Dockerfile is already configured to point the XLA builds to the expected gRPC port.
If no cache is running, XLA will print a warning but continue building without error.
However, the docker build must still have the host network mapped into the docker build
or the build will crash with a network error.

## Generate a development docker image

There is a `./build.py` script that starts a docker build with the correct
arguments for mapping the remote cache on the host network into the container.

```
legate-jax-workflows/docker $ ./build.py --framework maxtext
```

This will produce an image named `legate-jax-dev:maxtext` that can be tagged and pushed where needed.
The image will contain all the code from the repos and installs Python libraries in 
editable mode and enables rebuilds of the C++ libraries. To build without the remote cache,
use the `--no-cache` option to the build. For a full list of options see `./build.py --help`.
If you just need a base image for your devcontainer, make sure to specify 
`--stage devcontainer_base` to get an appropriate image.