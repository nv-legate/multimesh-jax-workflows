#! /usr/bin/env python

import sys
import argparse
import json
import os
import subprocess as sp
from pathlib import Path


import yaml
from subprocess_tee import run
from subprocess import CalledProcessError

try:
    from eos_workflows import (
        get_config,
        get_remote_config,
        get_worker_addr,
        save_maxtext_image,
        validate_maxtext_image,
    )

    user_config = get_config()
    repo_config_path = Path(__file__).parent.parent / "config.yml"
    with open(repo_config_path) as f:
        repo_config = yaml.safe_load(f)
except ImportError:
    pass

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--image",
    type=str,
    choices=["opt", "dev"],
    help="Whether to build a development image with all repos or a size-optimized release image",  # noqa: E501
    default="dev",
)

parser.add_argument(
    "--cache", action=argparse.BooleanOptionalAction, default=True
)  # noqa: E501

parser.add_argument(
    "--framework",
    type=str,
    choices=["maxtext"],
    default="maxtext",
)

parser.add_argument(
    "--commit",
    type=str,
    default=None,
    help="Commit the given container as an image instead of building",
)

parser.add_argument(
    "--dry-run",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to dry-run the commands without executing them",
)

parser.add_argument(
    "--save-remote",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to save the image on a remote filesystem",
)

parser.add_argument(
    "--build",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run a full docker build",
)

parser.add_argument(
    "--no-docker-build-cache",
    action="store_true",
    default=False,
    help="Do not use cache for docker build (clean build)",
)

parser.add_argument(
    "--remote",
    type=str,
    choices=["eos"],
    default="eos",
    help="The remote server to launch on",
)

parser.add_argument(
    "--validate",
    type=str,
    choices=["small", "medium", "large"],
    default=None,
    help="Run the specified validation job",
)

parser.add_argument(
    "--skip-save",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Skip the save sqsh step when running validation",
)

parser.add_argument(
    "--cache-addr",
    type=str,
    default="172.17.0.1",
    help="the port to use for the Bazel remote cache",
)

parser.add_argument(
    "--stage",
    type=str,
    default=None,
    help="the stage to build up to",
)

parser.add_argument(
    "--cuda-version",
    type=str,
    default="12.4.1",
    help="The CUDA version to use for the base CUDA image",
)

parser.add_argument(
    "--cudnn-version",
    type=str,
    default="",
    help="The cuDNN version to use for the base CUDA image",
)

parser.add_argument(
    "--build-type",
    type=str,
    choices=["Release", "Debug", "RelWithDebInfo"],
    default="Release",
    help="The type of CMake build to execute for Legate repos",
)

parser.add_argument(
    "--tag",
    type=str,
    default=None,
    help="The tag to use for naming the image",
)
parser.add_argument(
    "--latest", action=argparse.BooleanOptionalAction, default=False
)  # noqa: E501

parser.add_argument(
    "--repo",
    type=str,
    default="gitlab-master.nvidia.com:5005/legate/quickstart.internal",
    help="The repo (prefix) to use for image uploads",
)

parser.add_argument(
    "--upload", action=argparse.BooleanOptionalAction, default=False
)  # noqa: E501

args = parser.parse_args()

if args.image == "dev":
    dockerfile = "Dockerfile"
    short_image_name = "legate-jax-dev"
else:
    dockerfile = "Dockerfile.multi-stage"
    short_image_name = "legate-jax"

tag = args.tag or args.stage or args.framework
image_name = f"{short_image_name}:{tag}"

try:
    if args.commit:
        # first commit a temp image
        temp_image_name = f"{image_name}_temp"
        cmds = ["docker", "commit", args.commit, temp_image_name]
        run(cmds, check=True)

        # add the necessary workspace folders into the image
        cmds = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            "Dockerfile.commit",
            "--build-arg",
            f"COMMIT_IMAGE={temp_image_name}",
            ".",
        ]
        run(cmds, check=True)

        # remove the temp commit image
        cmds = ["docker", "rmi", "-f", temp_image_name]
        run(cmds, check=True)
    elif args.build:
        cmds = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            dockerfile,
            ".",
        ]
        if args.no_docker_build_cache:
            cmds.append("--no-cache")
        if args.stage:
            cmds.append("--target")
            cmds.append(args.stage)
        if args.cache:
            cmds = cmds + [
                "--network=host",
                "--add-host",
                f"host.docker.internal:{args.cache_addr}",
            ]
        else:
            cmds.append("--build-arg")
            cmds.append("BAZEL_CACHE=")

        for arg, value in (
            ("LEGATE_BUILD_TYPE", args.build_type),
            ("CUDA_VERSION", args.cuda_version),
            ("CUDNN_VERSION", args.cudnn_version),
            ("FRAMEWORK", args.framework),
        ):
            cmds.append("--build-arg")
            cmds.append(f"{arg}={value}")

        print(" ".join(cmds))

        output = run(cmds, check=True)

except CalledProcessError as cp:
    if cp.returncode != 0:
        sys.exit(cp.returncode)

remote_image = f"{args.repo}/{image_name}"
if args.upload:
    os.system(f"docker tag {image_name} {remote_image}")
    print(f"Pushing to {remote_image}")
    os.system(f"docker push {remote_image}")
    if args.latest:
        latest_image = f"{args.repo}/{short_image_name}:latest"
        print(f"Pushing to {latest_image}")
        os.system(f"docker push {latest_image}")

if args.save_remote:
    email = (
        sp.check_output(["git", "config", "--get", "user.email"])
        .decode("utf-8")
        .strip()
    )
    if args.framework == "maxtext":
        save_maxtext_image.remote(args.remote, tag=tag, email=email)

if args.validate:
    worker_addr = get_worker_addr(args.remote, user_config)
    remote_config = get_remote_config(args.remote, user_config)
    job_folder = remote_config.get("job_folder")
    image_folder = remote_config.get("image_folder")
    if job_folder is None or image_folder is None:
        raise Exception(
            "must specify both a job_folder and base_folder "
            f"for remote {args.remote} in config.yaml"
        )
    data = json.loads(
        sp.check_output(
            ["docker", "image", "ls", image_name, "--format", "json"]
        ).decode("utf-8")
    )
    image_id = data["ID"]
    email = (
        sp.check_output(["git", "config", "--get", "user.email"])
        .decode("utf-8")
        .strip()
    )

    maxtext_config = repo_config["maxtext"]
    job_config = maxtext_config["validate"][args.validate]

    result = validate_maxtext_image.remote(
        worker_addr,
        email=email,
        ID=image_id,
        tag=tag,
        dry_run=args.dry_run,
        skip_save=args.skip_save,
        job_folder=job_folder,
        image_folder=image_folder,
        **job_config,
    )
