#! /usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import re
import runpy
import subprocess as sp
import sys
from pathlib import Path

import yaml

try:
    from jax_plugins.multimesh import init
except ImportError:
    pass


parser = argparse.ArgumentParser(allow_abbrev=False)


# LEGION parameters
############################################
realm = parser.add_argument_group("Realm")
realm.add_argument(
    "--nodes",
    type=int,
    default=None,
    help="The number of nodes to run on",
)
realm.add_argument(
    "--network",
    type=str,
    choices=["none", "gasnetex", "ucx"],
    default="none",
    help="The Legion network module to use",
)
realm.add_argument(
    "--fbmem",
    type=int,
    default=70,
    help="The amount in GB of frame-buffer memory to use",
)
realm.add_argument(
    "--zcmem",
    type=int,
    default=4,
    help="The amount in GB of zero-copy (host pinned) memory to use",
)
realm.add_argument(
    "--sysmem",
    type=int,
    default=4,
    help="The amount in GB of host memory to use",
)
realm.add_argument(
    "--eager-sysmem",
    type=int,
    default=None,
    help="The amount in GB of host memory to reserve for eager allocations",  # noqa: E501
)
realm.add_argument(
    "--eager-fbmem",
    type=int,
    default=None,
    help="The amount in GB of frame-buffer memory to reserve for eager allocations",  # noqa: E501
)
realm.add_argument(
    "--gpus",
    type=int,
    default=8,
    help="The number of GPUs to use per-node.",
)
realm.add_argument(
    "--cpus",
    type=int,
    default=4,
    help="The number of CPUs to use per-node.",
)
realm.add_argument(
    "--profile",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="whether nsys profiling events should be created",  # noqa: E501
)


# XLA parameters
######################################
xla = parser.add_argument_group("XLA")
xla.add_argument(
    "--debug-nccl",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to print NCCL debug information",
)
xla.add_argument(
    "--use-nccl-comm-split",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to use comm split to create communicators",
)
xla.add_argument(
    "--collective-matmul",
    type=int,
    default=None,
    help="Specify the cutoff in MiB for activating collective matul/windowed einsum tensor parallelism",  # noqa: E501
)
xla.add_argument(
    "--hoist-loop-convert",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Wether to hoist converts inside a loop to avoid recomputation (at the cost of extra memory)",  # noqa: E501
)
xla.add_argument(
    "--latency-hiding-scheduler",
    action=argparse.BooleanOptionalAction,
    default=True,
)
xla.add_argument(
    "--xla-pipelining",
    action=argparse.BooleanOptionalAction,
    default=True,
)
xla.add_argument(
    "--xla-loop-buffering",
    action=argparse.BooleanOptionalAction,
    default=True,
)
xla.add_argument(
    "--xla-ar-threshold",
    type=int,
    default=4294967296,
)
xla.add_argument(
    "--xla-ag-threshold",
    type=int,
    default=4294967296,
)
xla.add_argument(
    "--xla-rs-threshold",
    type=int,
    default=33554432,
)


# MultiMesh parameters
####################################################
mm_jax = parser.add_argument_group("MultiMesh-Jax")
mm_jax.add_argument(
    "--backend",
    type=str,
    choices=["cuda", "multimesh", "cpu"],
    help="The JAX backend to use",
    default="multimesh",
)
mm_jax.add_argument(
    "--autoshard",
    action=argparse.BooleanOptionalAction,
    default=True,
)
mm_jax.add_argument(
    "--host-offload-min-reuse-distance",
    type=int,
    default=0,
    help="The minimum reuse distance to trigger host-offload of intermediates. 0 indicates no offloading",  # noqa: E501
)
mm_jax.add_argument(
    "--debug",
    type=str,
    default=None,
    choices=["none", "info", "debug", "spew"],
    help="The debug level",
)
mm_jax.add_argument(
    "--only-fuse-loop-tasks",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only fuse tasks inside loops",
)
mm_jax.add_argument(
    "--fuse-tasks",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Only fuse tasks inside loops",
)
xla_debug_levels = {
    None: 0,
    "info": 1,
    "debug": 3,
    "spew": 5,
}
mm_jax.add_argument(
    "--pp",
    type=int,
    default=1,
    help="The degree of pipeline parallelism",
)
mm_jax.add_argument(
    "--tp",
    type=int,
    default=1,
    help="The degree of tensor parallelism",
)
mm_jax.add_argument(
    "--fsdp",
    type=int,
    default=1,
    help="The degree of fully-sharded data parallelism",
)
mm_jax.add_argument(
    "--dp",
    type=int,
    default=1,
    help="The degree of data parallelism",
)
mm_jax.add_argument(
    "--interleave",
    type=int,
    default=1,
    help="The amount of interleaving (circular scheduling)",
)
mm_jax.add_argument(
    "--sequence-parallel",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use sequence parallelism",  # noqa: E501
)
mm_jax.add_argument(
    "--schedule",
    type=str,
    choices=["fill-drain", "gpipe", "1f1b", "wavefront", "prefetch-wavefront"],
    help="The microbatch schedule to use",
)
mm_jax.add_argument(
    "--microbatch-size",
    type=int,
    default=None,
    help="The size of the per-node microbatch to use. This is microbatch size per tensor-parallel domain, independent of data parallelism or FSDP. Default is to match the global batch size",  # noqa: E501
)
mm_jax.add_argument(
    "--hlo",
    type=str,
    default=None,
    help="Path to an HLO module to compile. This starts an HLO module compilation test rather than a full PaxML run",  # noqa: E501
)
mm_jax.add_argument(
    "--dump-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to only dump HLO modules without full execution",
)
mm_jax.add_argument(
    "--dump",
    type=str,
    default=None,
    help="A folder for dumping the HLO modules",
)
mm_jax.add_argument(
    "--dump-mpmd-passes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to dump all intermediate HLO modules from the MPMD passes",
)
mm_jax.add_argument(
    "--dump-all-passes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to dump all intermediate HLO modules",
)

mm_jax.add_argument(
    "--replicate-small-params",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Replicate all smaller than batch*squence_length",
)

# Maxtext parameters
####################################################
maxtext = parser.add_argument_group("MaxText")

maxtext.add_argument(
    "--maxtext-config",
    type=str,
    default=None,
    help="The path to the yaml config",
)

maxtext.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="Override model name in case base config is selected",
)

maxtext.add_argument(
    "--num-steps",
    type=int,
    default=None,
    help="The number of training steps to run",
)

maxtext.add_argument(
    "--num-layers",
    type=int,
    default=None,
    help="The number of transformer layers",
)

maxtext.add_argument(
    "--model-dims",
    type=int,
    default=None,
    help="The size of the model (embedding) dimension",
)

maxtext.add_argument(
    "--num-heads",
    type=int,
    default=None,
    help="The number of attention heads",
)

maxtext.add_argument(
    "--mlp-dims",
    type=int,
    default=None,
    help="The size of the model (embedding) dimension",
)

maxtext.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="The global batch size",
)

maxtext.add_argument(
    "--sequence-length",
    type=int,
    default=None,
    help="The maximum sequence length",
)

maxtext.add_argument(
    "--attention",
    type=str,
    default=None,
    choices=["autoselected", "dot_product", "flash", "cudnn_flash_te"],
    help="Override attention with one of valid attention types",
)

maxtext.add_argument(
    "--attention-type",
    type=str,
    default=None,
    choices=["global", "local_sliding"],
    help="Override attention-type  with one valid types",
)

maxtext.add_argument(
    "--remat",
    type=str,
    default=None,
    choices=[
        "minimal",
        "minimal_flash",
        "save_dot_except_mlpwi",
        "save_dot_except_mlp",
        "save_qkv_proj",
        "qkv_proj_offloaded",
        "minimal_offloaded",
        "save_out_proj",
        "full",
    ],
    help="Override remat_policy",
)

maxtext.add_argument(
    "--compile-topology-num-slices",
    type=int,
    default=None,
    help="Override slice topology",
)

maxtext.add_argument(
    "--use-iota-embed",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Trigger use_iota_embed, default is 'false'",
)

maxtext.add_argument(
    "--logits-dot-in-fp32",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Trigger logits_dot_in_fp32, default is 'false'",
)

maxtext.add_argument(
    "--scan-layers",
    dest="scan_layers",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to wrap repeated layers in a while-loop (True) or unroll (False)",  # noqa: E501
)


maxtext.add_argument(
    "--use-tfds-dataset",
    dest="use_tfds_dataset",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to use a TFDS dataset (True) or synthetic dataset (False)",
)

maxtext.add_argument(
    "--tokenizer-path",
    dest="tokenizer_path",
    default=None,
    help="Path to tokenizer file",
)

maxtext.add_argument(
    "--dataset-path",
    dest="dataset_path",
    default=None,
    help="Path to tfds dataset",
)

maxtext.add_argument(
    "--dataset-name",
    dest="dataset_name",
    default=None,
    help="Name of TFDS dataset to use",
)

############################################
args, realm_argv = parser.parse_known_args()

if args.scan_layers:
    raise Exception("--scan-layers is not yet support for MaxText")

if args.gpus > 0:
    if args.nodes > 1:
        hardware = "gpu_multiprocess"
    else:
        hardware = "gpu"
else:
    hardware = "cpu"

if args.dump_only:
    # forces a debug mode on the run where the HLO module
    # is generated from a single CPU run
    args.cpus = 1
    args.gpus = 0
    args.pp = 1
    args.tp = 1
    args.dp = 1
    args.fsdp = 1
    args.nodes = 1
    if args.batch_size is None:
        raise ValueError(
            "must give explicit --batch-size when using --dump-only"
        )  # noqa: E501


# setup environment variables
#############################
vmodule = [
    "mm_pjrt_buffer",
    "hlo_partition",
    "mm_computation",
    "mm_pjrt_client",
    "mm_pjrt_executable",
    "mpmd_input_output_buffer_alias",
    "loop_scheduler",
    "mm_ifrt_client",
    "hlo_memory_scheduler",
]

if args.debug_nccl:
    vmodule.append("nccl_utils")
    vmodule.append("nccl_collective_thunk")
    vmodule.append("nccl_api")

debug = None if args.debug == "none" else args.debug
xla_debug = xla_debug_levels[debug]

vmodule_str = ",".join([f"{root}={xla_debug}" for root in vmodule])
if custom_vmodule := os.environ.get("TF_CPP_VMODULE", None):
    vmodule_str = vmodule_str + "," + custom_vmodule

LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
env = dict(
    JAX_PLATFORMS=args.backend,
    TF_CPP_MIN_LOG_LEVEL=0,
    TF_CPP_MAX_LOG_LEVEL=xla_debug,
    TF_CPP_VMODULE=vmodule_str,
    JAX_TRACEBACK_FILTERING="off",
    JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0,
    ENABLE_TE=1,
    ENABLE_TE_SP=1,
    NVTE_FUSED_ATTN=1,
    LD_LIBRARY_PATH=f"{LD_LIBRARY_PATH}:/usr/local/cuda/lib64",
)

if args.backend == "multimesh":
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

xla_flags = [
    "--xla_disable_hlo_passes=rematerialization",
    f"--xla_gpu_enable_nccl_comm_splitting={str(args.use_nccl_comm_split).lower()}",  # noqa: E501
    f"--xla_force_host_platform_device_count={args.cpus}",
    f"--xla_gpu_enable_latency_hiding_scheduler={args.latency_hiding_scheduler}",  # noqa: E501
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_graph_level=0",
    f"--xla_gpu_all_reduce_combine_threshold_bytes={args.xla_ar_threshold}",  # noqa: E501
    f"--xla_gpu_all_gather_combine_threshold_bytes={args.xla_ag_threshold}",  # noqa: E501
    f"--xla_gpu_reduce_scatter_combine_threshold_bytes={args.xla_rs_threshold}",  # noqa: E501
    f"--xla_gpu_enable_pipelined_all_gather={args.xla_pipelining}",
    f"--xla_gpu_enable_pipelined_reduce_scatter={args.xla_pipelining}",
    f"--xla_gpu_enable_pipelined_all_reduce={args.xla_pipelining}",
    f"--xla_gpu_enable_while_loop_double_buffering={args.xla_loop_buffering}",  # noqa: E501
    "--xla_gpu_enable_all_gather_combine_by_dim=false",
    "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
]


# these come from the nvidia JAX toolbox benchmarks
if args.backend == "multimesh":
    xla_flags.append("--xla_gpu_enable_highest_priority_async_stream=true")

if args.collective_matmul is not None:
    xla_flags.extend(
        [
            f"--xla_gpu_threshold_for_windowed_einsum_mib={args.collective_matmul}",  # noqa: E501
            "--xla_gpu_multi_streamed_windowed_einsum=true",
            "--xla_gpu_use_memcpy_local_p2p=true",
        ]
    )

if args.hoist_loop_convert:
    os.environ["MULTIMESH_HOIST_CONVERT"] = "1"

if args.dump_only and args.dump is None:
    raise ValueError("--dump-only requsted, but no older passed to --dump")
if args.dump:
    xla_flags = xla_flags + [
        f"--xla_dump_to={args.dump}",
        "--xla_dump_hlo_as_text",
    ]
if args.dump_all_passes:
    xla_flags = xla_flags + [
        "--xla_dump_hlo_pass_re=.*",
    ]
elif args.dump_mpmd_passes:
    xla_flags = xla_flags + [
        "--xla_dump_hlo_pass_re=mpmd.*",
    ]


if existing_xla_flags := os.environ.get("XLA_FLAGS", None):
    xla_flags.append(existing_xla_flags)

env["XLA_FLAGS"] = " ".join(xla_flags)

# validate parallelism / device count
num_nodes = args.nodes or 1
total_parallelism = args.tp * args.pp * args.dp * args.fsdp
if args.gpus == 0:
    devices_per_node = args.cpus
else:
    devices_per_node = args.gpus
total_devices = devices_per_node * num_nodes

if total_parallelism != total_devices:
    raise ValueError(
        f"PP={args.pp} DP={args.dp} TP={args.tp} FSDP={args.fsdp} does not multiply to total no. of GPUS {total_devices}"  # noqa: E501
    )

# 2 perdevice if batch size is unspecified
batch_size = args.batch_size or total_devices * 2
mb_size = args.microbatch_size or batch_size
global_mb_size = mb_size * args.dp * args.fsdp
per_device_batch_size = batch_size // total_devices
devices_per_stage = total_devices // args.pp
transformer_num_devices = devices_per_stage
num_stages_per_interleave = args.pp
num_stages = num_stages_per_interleave * args.interleave
if args.num_layers is None:
    base_num_decoder_layers = None
    layers_per_stage = None
    layers_per_interleave = None
    if args.pp > 1 or batch_size != global_mb_size:
        raise Exception(
            "please specify --num-layers for configuring pipeline parallelism with --pp > 1 or microbatching"
        )
else:
    if args.num_layers % num_stages != 0:
        if args.num_layers < num_stages:
            error_str = f"Number of layers ({args.num_layers}) must be >= " \
                        f"product of pp ({args.pp}) and interleave ({args.interleave})"
        else:
            error_str = f"Number of layers ({args.num_layers}) must be divisible " \
                        f"by pp ({args.pp}) and interleave ({args.interleave})"
        raise ValueError(error_str)

    base_num_decoder_layers = args.num_layers
    layers_per_stage = base_num_decoder_layers // num_stages
    layers_per_interleave = base_num_decoder_layers // args.interleave

if batch_size % total_devices:
    raise ValueError(
        f"No support for partial batches, gpus={total_devices} "
        f"does not divide batch_size={batch_size}"
    )


if args.pp > 1:
    print(f"Pipeline parallelism with pp = {args.pp} enabled")

print(f"config = {args.maxtext_config}")
print(f"steps= {args.num_steps}")
print(f"per_device_batch_size = {per_device_batch_size}")
print(f"num_layers = {args.num_layers}")
print(f"max_target_length = {args.sequence_length}")
print(f"total_devices = {total_devices}")
print(f"devices_per_stage = {devices_per_stage}")
print(f"transformer_num_devices = {transformer_num_devices}")
print(f"num_stages_per_interleave = {num_stages_per_interleave}")
print(f"num_stages = {num_stages}")
print(f"batch_size = {batch_size}")
print(f"mb_size_per_node = {mb_size}")
print(f"global_mb_size = {global_mb_size}")


for key, val in env.items():
    os.environ[key] = str(val)
    print(f"env {key}={val}")

sys.path.append("/opt/maxtext/MaxText/")

import multimesh.jax  # noqa: E402

# initialize multimesh.jax
if args.backend == "multimesh":
    init(
        cpus=args.cpus,
        gpus=args.gpus,
        sysmem=args.sysmem * 1000,
        fbmem=args.fbmem * 1000,
        zcmem=args.zcmem * 1000,
        network=args.network,
        debug=debug,
        profile=args.profile,
        realm_argv=realm_argv,
    )

    devices = list(range(total_devices))

    transformer_axes = [
        ("data", "x"),
        ("stage", "y"),
        ("fsdp", "y"),
        ("fsdp_transpose", "y"),
    ]

    if args.sequence_parallel:
        transformer_axes.append(("sequence", "z"))
        transformer_axes.append(("tensor", "z"))
    else:
        transformer_axes.append(("tensor", "z"))
        transformer_axes.append(("sequence", "z"))

    transformer_axes += [
        ("autoregressive", "z"),
        ("data", "z"),
    ]

    transformer_x_dim = args.dp
    transformer_y_dim = args.fsdp
    transformer_z_dim = args.tp
    transformer_mesh = [
        transformer_x_dim,
        transformer_y_dim,
        transformer_z_dim,
    ]

    num_devices_for_all_loops = transformer_num_devices

    # register tasks
    import train
    from multimesh.jax import register_task

    if args.pp == 1 and global_mb_size == batch_size:
        # just default transformer parallelism
        register_task(
            "default",
            devices=devices[:transformer_num_devices],
            dims=transformer_mesh,
            device_axes=["x", "y", "z"],
            logical_axes=transformer_axes,
        )
    else:  # pp > 1 or microbatching
        train.set_mb_config(global_mb_size, args.schedule, num_stages, args.interleave)
        layer_regex = re.compile(r"layers_(\d+)")

        def compute_devices(name: str, backprop: bool):
            layer = int(layer_regex.search(name).groups()[0])
            pipeline_stage = layer // layers_per_stage
            if layers_per_interleave is not None:
                # layer offset within an interleave
                mesh = (layer % layers_per_interleave) // layers_per_stage
            else:
                mesh = pipeline_stage
            offset = transformer_num_devices * mesh
            stop = offset + transformer_num_devices
            suffix = "bwd" if backprop else "fwd"
            color = f"stage_{pipeline_stage}_{suffix}"
            return (offset, stop), color

        register_task(
            r"(layers_\d+)",
            callback=compute_devices,
            dims=transformer_mesh,
            device_axes=["x", "y", "z"],
            logical_axes=transformer_axes,
        )

import maxtext_utils as mu  # noqa: E402 must come after multimesh init

maxtext_base = Path(mu.__file__).parent

config = args.maxtext_config or maxtext_base / "configs" / "base.yml"

with open(config) as stream:
    model_yaml = yaml.safe_load(stream)
    keys = list(model_yaml.keys())
    keys.sort()
    for name in keys:
        print(name, model_yaml[name])

argv = [
    "this",
    str(config),
    "run_name=my_name",
    f"base_output_directory={os.getcwd()}/logs",
    "enable_single_controller=False",
    "enable_checkpointing=False",
    f"scan_layers={args.scan_layers}",
    f"per_device_batch_size={per_device_batch_size}",
    "monitor_goodput=False",
    "enable_goodput_recording=False",
    "enable_tensorboard=False",
]

if args.model_name is None:
    argv.append(f"use_iota_embed={args.use_iota_embed}")
    argv.append(f"logits_dot_in_fp32={args.logits_dot_in_fp32}")

if args.model_dims is not None:
    argv.append(f"base_emb_dim={args.model_dims}")
    mlp_dim = args.mlp_dims or args.model_dims * 4
else:
    mlp_dim = args.mlp_dims

if mlp_dim is not None:
    argv.append(f"base_mlp_dim={mlp_dim}")

if args.num_layers is not None:
    argv.append(f"base_num_decoder_layers={args.num_layers}")

if args.sequence_length is not None:
    argv.append(f"max_target_length={args.sequence_length}")

if args.num_heads is not None:
    argv.append(f"base_num_query_heads={args.num_heads}")
    argv.append(f"base_num_kv_heads={args.num_heads}")
    dims_per_head = args.model_dims // args.num_heads
    argv.append(f"head_dim={dims_per_head}")

if args.use_tfds_dataset:
    if args.dataset_path and args.dataset_name and args.tokenizer_path:
        argv.append("dataset_type=tfds")
        argv.append(f"dataset_path={args.dataset_path}")
        argv.append(f"dataset_name={args.dataset_name}")
        argv.append(f"tokenizer_path={args.tokenizer_path}")
    else:
        raise ValueError(
            "If --use-tfds-dataset is enabled, must define "
            "--tokenizer-path, --dataset-path, and --dataset-name"
        )
else:
    argv.append("dataset_type=synthetic")

argv.append(f"hardware={hardware}")

# parallelism has to be split betweeen the ICI and DCN
# explicitly for maxtext
num_local_devices = len(
    sp.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    .decode("utf-8")
    .splitlines()
)
if args.gpus == 0 or num_local_devices == 0:
    num_local_devices = args.cpus
total_nodes = total_devices // num_local_devices


def split_ici_dcn(agg_parallelism, p):
    if agg_parallelism >= num_local_devices:
        return (1, p, agg_parallelism * p)

    total_parallelism = p * agg_parallelism
    if total_parallelism <= num_local_devices:
        return (p, 1, agg_parallelism * p)

    dcn = total_parallelism // num_local_devices
    ici = p // dcn
    return (ici, dcn, agg_parallelism * p)


agg = 1  # aggregate parallelism
ici_tp, dcn_tp, agg = split_ici_dcn(agg, args.tp)
ici_pp, dcn_pp, agg = split_ici_dcn(agg, args.pp)
ici_fsdp, dcn_fsdp, agg = split_ici_dcn(agg, args.fsdp)
ici_dp, dcn_dp, agg = split_ici_dcn(agg, args.dp)

argv.append(f"ici_data_parallelism={ici_dp}")
argv.append(f"ici_fsdp_parallelism={ici_fsdp}")
argv.append(f"ici_pipeline_parallelism={ici_pp}")
argv.append(f"ici_tensor_parallelism={ici_tp}")
argv.append(f"dcn_data_parallelism={dcn_dp}")
argv.append(f"dcn_fsdp_parallelism={dcn_fsdp}")
argv.append(f"dcn_pipeline_parallelism={dcn_pp}")
argv.append(f"dcn_tensor_parallelism={dcn_tp}")

# optional maxtext args/overrides
if args.model_name:
    argv.append(f"model_name={args.model_name}")
if args.num_steps:
    argv.append(f"steps={args.num_steps}")
if args.attention:
    argv.append(f"attention={args.attention}")
if args.attention_type:
    argv.append(f"attention_type={args.attention_type}")
if args.remat:
    argv.append(f"remat_policy={args.remat}")
if args.compile_topology_num_slices:
    argv.append(
        f"compile_topology_num_slices={args.compile_topology_num_slices}"
    )  # noqa: E501


# always enable recomputation
with multimesh.jax.context(
    enable_recomputation=True,
    only_fuse_loop_tasks=args.only_fuse_loop_tasks,
    enable_task_fusion=args.fuse_tasks,
):
    if args.replicate_small_params:
        if args.sequence_length is None:
            raise Exception(
                "cannot set small parameter replication threshold without --sequence-length"  # noqa: E501
            )

        multimesh.jax.replicate_parameters_smaller_than_num_elements(
            batch_size * args.sequence_length
        )

    if args.hlo is None:
        import jaxlib

        # sys.argv = argv
        try:
            train.main(argv)
        except jaxlib.xla_extension.XlaRuntimeError as e:
            need_throw = True
            if args.dump_only:
                path = Path(args.dump)
                if path.exists():
                    globber = (
                        path / "*pjit_autoshard_step*before_optimizations.hlo.pb"
                    )  # noqa: E501
                    matches = glob.glob(str(globber))
                    if matches:
                        print(
                            "Dump seems to have succeeded in generating "
                            f"{matches[0]}. Run finished early with error: {e}"
                        )
                        need_throw = False

            if need_throw:
                raise e

    else:
        platform = "gpu" if args.gpus else "cpu"
        multimesh.jax.compile_hlo_module(
            args.hlo,
            num_partitions=total_devices,
            erase_sharding=args.erase_explicit_sharding,
            autoshard=args.autoshard,
            platform=platform,
            device_mem_gb=args.fbmem,
        )
