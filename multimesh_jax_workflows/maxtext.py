import numpy as np
from pathlib import Path
from typing import List, Optional

from eos_workflows import sbatch, srun
from eos_workflows import task
from eos_workflows.utils import chmod_x
from dask.distributed import print

from .parse_logs import (
    parse_jax_timings,
    parse_train_losses,
    parse_store_allocation_size_on_mesh,
    parse_flops,
)

def get_image_name(tag: str):
    return (
        f"gitlab-master.nvidia.com/legate/quickstart.internal/multimesh-jax-dev:{tag}"
    )


def bool_prefix(value: bool) -> str:
    if value:
        return ""
    return "no-"


def load_script_data(filename):
    with open(Path(__file__).parent.resolve() / "scripts" / filename) as f:
        return f.read()


SCRIPT_NAMES = [
    "dgx-h100.sh",
]

SCRIPTS = dict((script, load_script_data(script)) for script in SCRIPT_NAMES)


@task(output=str, jobid=str, returncode=int)
def run_maxtext(
    *,
    gpus: int,
    folder: str,
    model: str,
    image: str,
    dump: bool = False,
    launcher: str = "srun",
    tag: Optional[str] = None,
    platform: str = "dgx-h100",
    fbmem: int = 77,
    dp: int = 1,
    pp: int = 1,
    tp: int = 8,
    ep: int = 1,
    schedule: str = "prefetch-wavefront",
    num_layers: Optional[int] = None,
    num_steps: int = 20,
    eval_interval: int = -1,
    eval_steps: int = -1,
    dataset: str = "synthetic",
    tasks_per_node: Optional[int] = None,
    time: Optional[int] = None,
    dry_run: bool = False,
    email: Optional[str] = None,
    batch_size: Optional[int] = None,
    microbatch_size: Optional[int] = 4,
    sequence_length: Optional[int] = None,
    capacity_factor: float = -1,
    hoist_loop_convert: bool = False,
):
    if launcher == "srun":
        launcher = srun
    else:
        launcher = sbatch

    LAYERS = {
        "gpt3-24layers": 24,
        "gpt3-175b": 96,
        "llama3-8b": 32,
        "llama3-70b": 80,
        "mixtral-8x7b": 32,
    }
    num_layers = num_layers or LAYERS[model]

    SEQ = {
        "gpt3-24layers": 2048,
        "gpt3-175b": 2048,
        "llama3-8b": 4096,
        "llama3-70b": 4096,
        "mixtral-8x7b": 4096,
    }
    sequence_length = sequence_length or SEQ[model]

    assert dp * pp * tp * ep == gpus
    assert num_layers % pp == 0
    if platform.lower() == "dgx-h100":
        gpus_per_node = 8
        num_nodes = gpus // gpus_per_node
        tasks_per_node = tasks_per_node or 8
    tasks_per_node = tasks_per_node or 1
    interleave = num_layers // pp

    batch_size = batch_size or gpus * 2

    if time is None or time == 0:
        # budget 15 seconds per step + 7 mins to init
        time = max(20, int(num_steps * 0.25) + 7)

    DS_CONFIG = {
        "c4": {
            "eos_path": "/lustre/share/coreai_dlalgo_ci/artifacts/text/c4",
            "mount_path": "/data/c4:ro",  # including ro tag here
            "dataset_name": "c4/en:3.1.0",
            "dataset_path": "/data/c4/mlperf-tfrecord",
            "tokenizer_path": "/opt/maxtext/assets/tokenizer.llama2",
        }
    }

    if dataset != "synthetic":
        assert dataset in DS_CONFIG, f"Dataset {dataset} not supported"
        config = DS_CONFIG[dataset]

        use_tfds_dataset = True
        dataset_name = config["dataset_name"]
        dataset_path = config["dataset_path"]
        tokenizer_path = config["tokenizer_path"]

        dataset_mounts = {config["eos_path"]: config["mount_path"]}
    else:
        use_tfds_dataset = False
        dataset_name, dataset_path, tokenizer_path = None, None, None
        dataset_mounts = {}

    if eval_interval > 0 and eval_steps > 0:
        perform_eval = True
        eval_batch_size = batch_size
    elif eval_interval > 0 or eval_steps > 0:
        raise ValueError(
            "Both eval_interval and eval_steps must be set > 0 to perform eval"
        )
    else:
        perform_eval = False
        eval_batch_size = 0

    if dump:
        dump_flags = "--dump --dump-mpmd-passes"
    else:
        dump_flags = ""

    print(f"Running {model} in", folder)
    folder = Path(folder)
    script = SCRIPTS[f"{platform}.sh"].format(
        fbmem=fbmem,
        num_layers=num_layers,
        model=model,
        sequence_length=sequence_length,
        capacity_factor=capacity_factor,
        batch_size=batch_size,
        microbatch_size=microbatch_size,
        interleave=interleave,
        num_steps=num_steps,
        use_tfds_dataset=bool_prefix(use_tfds_dataset),
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer_path=tokenizer_path,
        perform_eval=bool_prefix(perform_eval),
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        eval_batch_size=eval_batch_size,
        dump_flags=dump_flags,
        pp=pp,
        dp=dp,
        tp=tp,
        ep=ep,
        schedule=schedule,
        num_gpus=gpus,
        num_nodes=num_nodes,
        hoist_loop_convert=bool_prefix(hoist_loop_convert),
    )

    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "local.sh", "w") as f:
        f.write(script)
    chmod_x(folder / "local.sh")

    cmd = ["/opt/entrypoint.sh", "/workspace/cwd/local.sh"]
    extra_flags = [
        "--no-container-mount-home",
        "--no-container-entrypoint",
    ]

    mounts = {
        folder: "/workspace/cwd",
    }
    mounts.update(dataset_mounts)

    env = dict(
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64",
    )
    return launcher(
        *extra_flags,
        image=image,
        folder=folder,
        cmd=cmd,
        application="maxtext",
        time=time,
        dry_run=dry_run,
        mounts=mounts,
        email=email,
        tasks_per_node=tasks_per_node,
        tag=tag,
        mpi="pmix",
        env=env,
        N=num_nodes,
    )


@task(passed=bool)
def validate_timings(
    *, timings: List[float], max_mean: float, max_stdev: float, model: str = ""
):
    arr = np.array(timings)
    mean = np.mean(arr)
    stdev = np.std(arr)

    if mean <= max_mean:
        mean_ok = True
        mean_comp = "<="
    else:
        mean_ok = False
        mean_comp = ">"

    if stdev <= max_stdev:
        std_ok = True
        std_comp = "<="
    else:
        std_ok = False
        std_comp = ">"

    if mean_ok and std_ok:
        result = True
        msg = "PASS"
    else:
        msg = "FAIL"
        result = False

    print(
        f"{msg}: {model:16}"
        f" mean {mean:8.4f} {mean_comp} {max_mean:<8.4f},"
        f" std {stdev:8.4f} {std_comp} {max_stdev:<8.4f}"
    )
    return result


@task(timings=List[float], losses=List[float], memory=int, flops=float)
def parse_maxtext_results(
    job_folder: str,
    dry_run: bool = False,
):
    timings = parse_jax_timings(
        folder=job_folder, logfile="0/jax_0.log", dry_run=dry_run
    )
    losses = parse_train_losses(folder=job_folder, logfile="0/0.out", dry_run=dry_run)
    memory = parse_store_allocation_size_on_mesh(
        folder=job_folder,
        logfile="1/jax_1.log",
        dry_run=dry_run,
        device=0,
    )
    flops = parse_flops(folder=job_folder, logfile="0/0.out", dry_run=dry_run)
    return timings, losses, memory, flops


@task(
    timings=List[float],
    losses=List[float],
    memory=int,
    flops=float,
    jobid=str,
    rc=int,
)
def get_maxtext_results(
    *,
    image: str,
    job_folder: str,
    model: str,
    gpus: int,
    platform: str = "dgx-h100",
    dry_run: bool = False,
    email: Optional[str] = None,
    **maxtext_kwargs,
):
    output, jobid, rc = run_maxtext.defer(
        image=image,
        folder=job_folder,
        gpus=gpus,
        model=model,
        platform=platform,
        dry_run=dry_run,
        email=email,
        **maxtext_kwargs,
    )

    timings, losses, memory, flops = parse_maxtext_results.invoke(
        job_folder=job_folder, wait_on=[output], dry_run=dry_run
    )

    return timings, losses, memory, flops, jobid, rc
