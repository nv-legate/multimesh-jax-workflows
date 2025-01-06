#! /usr/bin/env python

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
    from jax_plugins.legate import init
except ImportError:
    pass


parser = argparse.ArgumentParser(allow_abbrev=False)


# LEGION parameters
############################################
legion = parser.add_argument_group("Legion")
legion.add_argument(
    "--nodes",
    type=int,
    default=None,
    help="The number of nodes to run on",
)
legion.add_argument(
    "--network",
    type=str,
    choices=["none", "gasnetex", "ucx"],
    default="none",
    help="The Legion network module to use",
)
legion.add_argument(
    "--fbmem",
    type=int,
    default=70,
    help="The amount in GB of frame-buffer memory to use",
)
legion.add_argument(
    "--zcmem",
    type=int,
    default=4,
    help="The amount in GB of zero-copy (host pinned) memory to use",
)
legion.add_argument(
    "--sysmem",
    type=int,
    default=4,
    help="The amount in GB of host memory to use",
)
legion.add_argument(
    "--eager-sysmem",
    type=int,
    default=None,
    help="The amount in GB of host memory to reserve for eager allocations",  # noqa: E501
)
legion.add_argument(
    "--eager-fbmem",
    type=int,
    default=None,
    help="The amount in GB of frame-buffer memory to reserve for eager allocations",  # noqa: E501
)
legion.add_argument(
    "--gpus",
    type=int,
    default=8,
    help="The number of GPUs to use per-node.",
)
legion.add_argument(
    "--cpus",
    type=int,
    default=4,
    help="The number of CPUs to use per-node.",
)
legion.add_argument(
    "--profile",
    type=str,
    default=None,
    help="The root of the profile file, if Legion profiling should be activated",  # noqa: E501
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


# XLA parameters
####################################################
legate_jax = parser.add_argument_group("Legate-Jax")
legate_jax.add_argument(
    "--backend",
    type=str,
    choices=["cuda", "legate", "cpu"],
    help="The JAX backend to use",
    default="legate",
)
legate_jax.add_argument(
    "--autoshard",
    action=argparse.BooleanOptionalAction,
    default=True,
)
legate_jax.add_argument(
    "--host-offload-min-reuse-distance",
    type=int,
    default=0,
    help="The minimum reuse distance to trigger host-offload of intermediates. 0 indicates no offloading",  # noqa: E501
)
legate_jax.add_argument(
    "--debug",
    type=str,
    default=None,
    choices=["info", "debug", "spew"],
    help="The debug level",
)
legate_jax.add_argument(
    "--only-fuse-loop-tasks",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only fuse tasks inside loops",
)
legate_jax.add_argument(
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
legate_jax.add_argument(
    "--pp",
    type=int,
    default=1,
    help="The degree of pipeline parallelism",
)
legate_jax.add_argument(
    "--tp",
    type=int,
    default=1,
    help="The degree of tensor parallelism",
)
legate_jax.add_argument(
    "--fsdp",
    type=int,
    default=1,
    help="The degree of fully-sharded data parallelism",
)
legate_jax.add_argument(
    "--dp",
    type=int,
    default=1,
    help="The degree of data parallelism",
)
legate_jax.add_argument(
    "--interleave",
    type=int,
    default=1,
    help="The amount of interleaving (circular scheduling)",
)
legate_jax.add_argument(
    "--sequence-parallel",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to use sequence parallelism",  # noqa: E501
)
legate_jax.add_argument(
    "--schedule",
    type=str,
    choices=["fill-drain", "gpipe", "1f1b", "wavefront"],
    help="The microbatch schedule to use",
)
legate_jax.add_argument(
    "--microbatch-size",
    type=int,
    default=None,
    help="The size of the microbatches to use. Default is to match the global batch size",  # noqa: E501
)
legate_jax.add_argument(
    "--hlo",
    type=str,
    default=None,
    help="Path to an HLO module to compile. This starts an HLO module compilation test rather than a full PaxML run",  # noqa: E501
)
legate_jax.add_argument(
    "--dump-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to only dump HLO modules without full execution",
)
legate_jax.add_argument(
    "--dump",
    type=str,
    default=None,
    help="A folder for dumping the HLO modules",
)
legate_jax.add_argument(
    "--dump-all-passes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to dump all intermediate HLO modules",
)

legate_jax.add_argument(
    "--replicate-small-params",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Replicate all smaller than batch*squence_length",
)

legate_jax.add_argument(
    "--load-balance-embeddings",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to rotate microbatches across different submeshes for load-balancing",  # noqa: E501
)

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

############################################
args, realm_argv = parser.parse_known_args()

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
        )


# setup environment variables
#############################
vmodule = [
    "legate_pjrt_buffer",
    "hlo_partition",
    "legate_computation",
    "legate_pjrt_client",
    "legate_pjrt_executable",
    "mpmd_input_output_buffer_alias",
    "legate_store_cache",
    "loop_scheduler",
    "legate_ifrt_client",
    "hlo_memory_scheduler",
]

if args.debug_nccl:
    vmodule.append("nccl_utils")
    vmodule.append("nccl_collective_thunk")
    vmodule.append("nccl_api")

xla_debug = xla_debug_levels[args.debug]

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
    LD_LIBRARY_PATH=f"{LD_LIBRARY_PATH}:/usr/local/cuda/lib64",
)

if args.backend == "legate":
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

xla_flags = [
    "--xla_disable_hlo_passes=rematerialization",
    f"--xla_gpu_enable_nccl_comm_splitting={str(args.use_nccl_comm_split).lower()}",  # noqa: E501
    f"--xla_force_host_platform_device_count={args.cpus}",
    f"--xla_gpu_enable_latency_hiding_scheduler={args.latency_hiding_scheduler}",  # noqa: E501
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_graph_level=0",
]


# these come from the nvidia JAX toolbox benchmarks
if args.backend == "legate":
    xla_flags.extend(
        [
            f"--xla_gpu_all_reduce_combine_threshold_bytes={args.xla_ar_threshold}",  # noqa: E501
            f"--xla_gpu_all_gather_combine_threshold_bytes={args.xla_ag_threshold}",  # noqa: E501
            f"--xla_gpu_reduce_scatter_combine_threshold_bytes={args.xla_rs_threshold}",  # noqa: E501
            f"--xla_gpu_enable_pipelined_all_gather={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_reduce_scatter={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_all_reduce={args.xla_pipelining}",
            f"--xla_gpu_enable_while_loop_double_buffering={args.xla_loop_buffering}",  # noqa: E501
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            # these are specific to legate
            "--xla_gpu_enable_highest_priority_async_stream=true",
            "--xla_gpu_enable_triton_softmax_fusion=false",
            # "--xla_gpu_all_reduce_combine_threshold_bytes=51200",
            # "--xla_dump_hlo_pass_re=.*",
        ]
    )
else:
    xla_flags.extend(
        [
            f"--xla_gpu_all_reduce_combine_threshold_bytes={args.xla_ar_threshold}",  # noqa: E501
            f"--xla_gpu_all_gather_combine_threshold_bytes={args.xla_ag_threshold}",  # noqa: E501
            f"--xla_gpu_reduce_scatter_combine_threshold_bytes={args.xla_rs_threshold}",  # noqa: E501
            f"--xla_gpu_enable_pipelined_all_gather={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_reduce_scatter={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_all_reduce={args.xla_pipelining}",
            f"--xla_gpu_enable_while_loop_double_buffering={args.xla_loop_buffering}",  # noqa: E501
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            "--xla_gpu_enable_triton_softmax_fusion=false",
        ]
    )


if args.collective_matmul is not None:
    xla_flags.extend(
        [
            f"--xla_gpu_threshold_for_windowed_einsum_mib={args.collective_matmul}",  # noqa: E501
            "--xla_gpu_multi_streamed_windowed_einsum=true",
            "--xla_gpu_use_memcpy_local_p2p=true",
        ]
    )


if args.dump_only and args.dump is None:
    raise ValueError(
        "--dump-only requsted, but not HLO dump folder passed to --dump"
    )
if args.dump:
    xla_flags = xla_flags + [
        f"--xla_dump_to={args.dump}",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_as_proto",
    ]
if args.dump_all_passes:
    xla_flags = xla_flags + [
        "--xla_dump_hlo_pass_re=.*",
    ]

if existing_xla_flags := os.environ.get("XLA_FLAGS", None):
    xla_flags.append(existing_xla_flags)

env["XLA_FLAGS"] = " ".join(xla_flags)

# valiate parallelism / device count
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

# 2 perdevice if batch size is specified
batch_size = args.batch_size or total_devices * 2
mb_size = args.microbatch_size or batch_size
per_device_batch_size = batch_size // total_devices
devices_per_stage = total_devices // args.pp
transformer_num_devices = devices_per_stage
num_stages_per_interleave = args.pp
num_stages = num_stages_per_interleave * args.interleave
# 8 layers given by smoke test
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
print(f"mb_size = {mb_size}")


for key, val in env.items():
    os.environ[key] = str(val)
    print(f"env {key}={val}")

sys.path.append("/opt/maxtext/MaxText/")

import legate.jax  # noqa: E402

# initialize legate.jax
if args.backend == "legate":
    init(
        cpus=args.cpus,
        gpus=args.gpus,
        sysmem=args.sysmem * 1000,
        fbmem=args.fbmem * 1000,
        zcmem=args.zcmem * 1000,
        network=args.network,
        debug=args.debug,
        profile=args.profile,
        realm_argv=realm_argv,
    )

    devices = list(range(total_devices))

    transformer_axes = [
        ("data", "x"),
        ("stage", "y"),
        ("fsdp", "y"),
        ("fsdp_transpose", "y"),
        ("sequence", "z"),
        ("tensor", "z"),
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

    if args.load_balance_embeddings:
        loop_dependent_submesh_size = transformer_num_devices
        num_devices_for_all_loops = total_devices
    else:
        loop_dependent_submesh_size = None
        num_devices_for_all_loops = transformer_num_devices

    # register tasks
    import train
    from legate.jax import register_task

    if args.pp == 1 and mb_size == batch_size:
        # just default transformer parallelism
        register_task(
            "default",
            devices=devices[:transformer_num_devices],
            dims=transformer_mesh,
            device_axes=["x", "y", "z"],
            logical_axes=transformer_axes,
        )
    else:  # pp > 1 or microbatching
        train.set_mb_config(
            mb_size, args.schedule, num_stages, args.interleave
        )
        layer_regex = re.compile(r"layers_(\d+)")

        def compute_devices(name: str):
            layer = int(layer_regex.search(name).groups()[0])
            if layers_per_interleave is not None:
                # layer offset within an interleave
                layer = layer % layers_per_interleave
            stage = layer // layers_per_stage
            offset = transformer_num_devices * stage
            stop = offset + transformer_num_devices
            return list(range(offset, stop))

        register_task(
            r"(layers_\d+)",
            callback=compute_devices,
            dims=transformer_mesh,
            device_axes=["x", "y", "z"],
            logical_axes=transformer_axes,
        )

        first_layer_devices = devices[:num_devices_for_all_loops]
        last_layer_devices = devices[-num_devices_for_all_loops:]

        layer_meshes = {
            "(emb_lookup).*": (
                first_layer_devices,
                loop_dependent_submesh_size,
                False,
            ),
            "(position_emb).*": (
                first_layer_devices,
                loop_dependent_submesh_size,
                False,
            ),
            "(decoder_norm).*": (
                last_layer_devices,
                loop_dependent_submesh_size,
                True,
            ),
            "(compute_loss).*": (
                last_layer_devices,
                loop_dependent_submesh_size,
                True,
            ),
            "(final_ln).*": (
                last_layer_devices,
                loop_dependent_submesh_size,
                True,
            ),
            "default": (devices[:transformer_num_devices], None, False),
        }

        for layer, (
            devices,
            loop_submesh_size,
            submesh_rotation,
        ) in layer_meshes.items():
            register_task(
                layer,
                devices=devices,
                dims=transformer_mesh,
                device_axes=["x", "y", "z"],
                logical_axes=transformer_axes,
                loop_submesh_size=loop_submesh_size,
                loop_submesh_reverse=submesh_rotation,
            )

import maxtext_utils as mu  # noqa: E402 must come after legate init

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
    f"base_output_directory=${os.getcwd()}/logs",
    "dataset_type=synthetic",
    "enable_single_controller=False",
    "enable_checkpointing=False",
    f"scan_layers={args.scan_layers}",
    f"per_device_batch_size={per_device_batch_size}",
    f"use_iota_embed={args.use_iota_embed}",
    f"logits_dot_in_fp32={args.logits_dot_in_fp32}",
]

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

argv.append(f"hardware={hardware}")

# parallelism has to be split betweeen the ICI and DCN
# explicitly for maxtext
num_local_devices = len(
    sp.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    .decode("utf-8")
    .splitlines()
)
if args.gpus == 0 or num_local_devices == 0:
    num_local_devices = args.cpus
total_nodes = total_devices // num_local_devices
dcn_pp = total_nodes
ici_pp = args.pp // total_nodes

argv.append(f"ici_data_parallelism={args.dp}")
argv.append(f"ici_tensor_parallelism={args.tp}")
argv.append(f"ici_pipeline_parallelism={ici_pp}")
argv.append(f"dcn_pipeline_parallelism={dcn_pp}")
argv.append(f"ici_fsdp_parallelism={args.fsdp}")

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
    )


# always enable recomputation
with legate.jax.context(
    enable_recomputation=True,
    host_offload_min_reuse_distance=args.host_offload_min_reuse_distance,
    only_fuse_loop_tasks=args.only_fuse_loop_tasks,
    enable_task_fusion=args.fuse_tasks,
):
    if args.replicate_small_params:
        if args.sequence_length is None:
            raise Exception(
                "cannot set small parameter replication threshold without --sequence-length"  # noqa: E501
            )

        legate.jax.replicate_parameters_smaller_than_num_elements(
            batch_size * args.sequence_length
        )

    if args.hlo is None:
        import jaxlib

        sys.argv = argv
        try:
            if args.autoshard:
                # invoke directly to preserve microbatch settings
                train.main(argv)
            else:
                runpy.run_path(
                    "/opt/maxtext/MaxText/train.py", run_name="__main__"
                )
        except jaxlib.xla_extension.XlaRuntimeError as e:
            need_throw = True
            if args.dump_only:
                path = Path(args.dump)
                if path.exists():
                    globber = (
                        path
                        / "*pjit_autoshard_step*before_optimizations.hlo.pb"
                    )
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
        legate.jax.compile_hlo_module(
            args.hlo,
            num_partitions=total_devices,
            erase_sharding=args.erase_explicit_sharding,
            autoshard=args.autoshard,
            platform=platform,
            device_mem_gb=args.fbmem,
        )
