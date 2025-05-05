# For Developers

## Getting Started

This repository is the base development project for vscode workflows.
To get started, open the repo folder in vscode. On the first time
starting a devcontainer, do the following:

* Install the Docker extension in your vscode
* Create a conda environment, e.g. `dev-workflows` from the provided environment file on your development machine

```bash
multimesh-jax-workflows $ conda env create -n dev-workflows -f environment.yml
```

* Install all Python dependencies.

```bash
multimesh-jax-workflows $ conda activate dev-workflows
multimesh-jax-workflows $ python -m pip install -r requirements.txt
```

* Create a conda environment from `environment.yml` on all remote machines you want to use for testing
* Create a `~/.config/workflows/config.yml` defining parameters for each remote you want to use for testing

Inside the project, the following steps are then required for each new development environment:

## Creating feature-specific devcontainers

The easiest way is to use the pre-configured task `New Workspace In Folder' (shift+cmd+b on Mac)`.
The user is then prompted to give the image name and a unique name for the container.
The chosen name should describe the feature you are going to develop.
A code window should open with a fresh monorepo checkout.

## clangd and compile_comands.json for XLA

The main recommended extension is `clangd`. The container comes with a
prebuilt `compile_commands.json` using the hedronvision compile commands
extractor.

## Development Tasks

Most relevent development tasks can be run as vscode tasks.
Tasks are generally either build/test tasks run *inside* the container
or upload/run tasks run *on* the image/container.

### Tasks inside the container

The main tasks inside the container are build and test tasks:

* Build multimesh-jax plugin
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
