#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import sys
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional

import yaml
from subprocess_tee import run

try:
    from eos_workflows import get_config, save_sqsh_image

    from multimesh_jax_workflows import run_multimesh_testsuite

    user_config = get_config()
    repo_config_path = Path(__file__).parent.parent / "config.yml"
    with open(repo_config_path) as f:
        repo_config = yaml.safe_load(f)
except ImportError:
    pass


def check_arg_None(arg: str) -> Optional[str]:
    return None if arg == "None" else arg


parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--cache", action=argparse.BooleanOptionalAction, default=True
)  # noqa: E501

parser.add_argument(
    "--framework",
    type=str,
    default="maxtext",
    choices=["maxtext"],
)

parser.add_argument(
    "--commit",
    type=str,
    default=None,
    help="Commit the given container as an image instead of building",
)

parser.add_argument(
    "--base-component",
    type=str,
    choices=["realm", "zuku", "multimesh", "cuda", "None"],
    default=None,
    help="The component to start building from",
)

parser.add_argument(
    "--base-image",
    type=str,
    default=None,
    help="The base image to use for the build",
)

parser.add_argument(
    "--dry-run",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to dry-run the commands without executing them",
)

parser.add_argument(
    "--include-nsys",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to include nsys in the final container build",
)

parser.add_argument(
    "--nsys-url",
    type=str,
    default="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/nsight-systems-2025.3.1_2025.3.1.90-1_amd64.deb",  # noqa: E501
    help="The download location for the nsys deb file",
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
    default=None,
    choices=["small", "medium", "large", "moe", "convergence", "zero-bubble", "all"],
    help="Run a validation job on the generated image",
)

parser.add_argument(
    "--sqsh",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="save a sqsh image",
)

parser.add_argument(
    "--sqsh-before-validate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="save a sqsh image before validating",
)

parser.add_argument(
    "--cache-addr",
    type=str,
    default="172.17.0.1",
    help="the port to use for the Bazel remote cache",
)

parser.add_argument(
    "--include-source", action=argparse.BooleanOptionalAction, default=False
)

parser.add_argument(
    "--stage",
    type=str,
    default="final_image",
    help="the stage to build up to",
)

parser.add_argument(
    "--cuda-version",
    type=str,
    default="12.8.1",
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
    help="The type of CMake build to execute for repos",
)

parser.add_argument(
    "--tag",
    type=str,
    default=None,
    help="The tag to use for naming the image",
)

parser.add_argument(
    "--repo",
    type=str,
    default="gitlab-master.nvidia.com:5005/legate/quickstart.internal",
    help="The repo (prefix) to use for image uploads",
)

parser.add_argument(
    "--name",
    type=str,
    default="multimesh-jax-dev",
    help="The name of the image: <name>:<tag>",
)  # noqa: E501

parser.add_argument(
    "--upload", action=argparse.BooleanOptionalAction, default=False
)  # noqa: E501

args = parser.parse_args()

args.base_image = check_arg_None(args.base_image)
args.base_component = check_arg_None(args.base_component)

dockerfile = "Dockerfile"
short_image_name = args.name


def add_nsys_args(cmds, args):
    if args.include_nsys:
        cmds.append("--build-arg")
        cmds.append("INCLUDE_NSYS=nsys")
        deb_name = Path(args.nsys_url).name
        cmds.append("--build-arg")
        cmds.append(f"NSYS_URL={args.nsys_url}")
        cmds.append("--build-arg")
        cmds.append(f"NSYS_DEB={deb_name}")
        cmds.append("--network=host")


tag = args.tag or args.commit or args.stage or args.framework
image_name = f"{short_image_name}:{tag}"

try:
    if args.commit:
        # first commit a temp image
        temp_image_name = f"{image_name}_temp"
        cmds = ["docker", "commit", args.commit, temp_image_name]
        if not args.dry_run:
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
        add_nsys_args(cmds, args)
        if not args.dry_run:
            run(cmds, check=True)

        # remove the temp commit image
        cmds = ["docker", "rmi", "-f", temp_image_name]
        if not args.dry_run:
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

        cmds.append("--build-arg")
        if args.include_source:
            cmds.append("FINAL_IMAGE_BASE=distribute")
        else:
            cmds.append("FINAL_IMAGE_BASE=install_tools")

        for arg, value in (
            ("BUILD_TYPE", args.build_type),
            ("CUDA_VERSION", args.cuda_version),
            ("CUDNN_VERSION", args.cudnn_version),
            ("FRAMEWORK", args.framework),
        ):
            cmds.append("--build-arg")
            cmds.append(f"{arg}={value}")

        if args.base_image:
            if args.base_component is None:
                raise ValueError(
                    "must give a --base-component for the --base-image"
                )  # noqa: E501
            cmds.append("--build-arg")
            cmds.append(
                f"{args.base_component.upper()}_BASE_IMAGE={args.base_image}"
            )  # noqa: E501

        add_nsys_args(cmds, args)
        print(" ".join(cmds))

        if not args.dry_run:
            output = run(cmds, check=True)

except CalledProcessError as cp:
    if cp.returncode != 0:
        sys.exit(cp.returncode)

if (args.validate or args.upload) and args.stage == "final_image" and not args.dry_run:
    # make sure the image passes smoke test before pushing
    cmd = [
        "docker",
        "run",
        "--entrypoint",
        "/opt/entrypoint.sh",
        image_name,
        "python",
        "-c",
        "import multimesh.jax",
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise Exception(f"smoke test failed for image {image_name}")

if args.repo:
    remote_image = f"{args.repo}/{image_name}"
    os.system(f"docker tag {image_name} {remote_image}")

if args.upload or args.validate or args.sqsh:
    if args.repo is None:
        raise Exception("upload to remote repo requested, but no --repo specified")
    print(f"Pushing to {remote_image}")
    if not args.dry_run:
        os.system(f"docker push {remote_image}")

if args.sqsh:
    remote_read_image = re.compile(r":\d+").sub("", remote_image)
    result = save_sqsh_image.remote(
        remote=args.remote, dry_run=args.dry_run, image=remote_read_image
    )
    print(f"Saved sqsh file to {result} on {args.remote}")

if args.validate:
    name = args.tag or args.commit or args.stage
    tag = None if args.validate == "all" else args.validate
    # remove port numbers when reading the image name
    remote_read_image = re.compile(r":\d+").sub("", remote_image)
    result = run_multimesh_testsuite(
        name=name,
        image=remote_read_image,
        dry_run=args.dry_run,
        save_image=args.sqsh_before_validate,
        tag=tag,
        perf_only=True,
    )
    print(result.data)
