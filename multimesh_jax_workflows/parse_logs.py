from eos_workflows import task, get_file_text
from typing import List
from pathlib import Path
import re
from dask.distributed import print


@task(timings=List[float])
def parse_jax_timings(
    *,
    folder: str,
    logfile: str = "0/jax_0.log",
    name: str = "autoshard",
    dry_run: bool = False,
):
    if dry_run:
        print("Dry run, returning dummy timings...")
        return [0.0, 0.0, 0.0, 0.0]
    timing_re = re.compile(f"{name}.*?finished in (\d+[.]\d+)")
    log_path = Path(folder) / logfile
    text = get_file_text(log_path)
    if text is None:
        return []
    return list(map(float, timing_re.findall(text)))


@task(timings=List[float])
def parse_train_losses(
    *,
    folder: str,
    logfile: str = "0/0.out",
    dry_run: bool = False,
):
    if dry_run:
        print("Dry run, returning dummy losses...")
        return [0.0, 0.0, 0.0, 0.0]
    loss_re = re.compile(r"completed step:.*loss: (\d+[.]\d+)")
    log_path = Path(folder) / logfile
    text = get_file_text(log_path)
    if text is None:
        return []
    return list(map(float, loss_re.findall(text)))


@task(memory=int)
def parse_store_allocation_size_on_mesh(
    folder: str | Path, logfile: str, device: int, dry_run=False
):
    if dry_run:
        return 100

    log_path = Path(folder) / logfile
    text = get_file_text(log_path)
    if text is None:
        return 0
    all_stores = re.compile("CreateStore(.*)").findall(text)
    size_re = re.compile(r"size=(\d+)")
    shard_re = re.compile(r"shard=(\d+)")
    type_re = re.compile(r"type=(\d+)")
    devices_re = re.compile(r"devices.(\d+)...(\d+)")
    total_allocated = 0
    for line in all_stores:
        typ = int(type_re.search(line).groups()[0])
        sizes = size_re.findall(line)
        shards = shard_re.findall(line)
        shard_size = 1
        start, stop = map(int, devices_re.search(line).groups())
        if device >= start and device < stop:
            for size, sharding in zip(sizes, shards):
                size = int(size)
                sharding = int(sharding)
                if size >= sharding:
                    shard_size *= size / sharding

            if typ == 2:
                shard_size *= 2
            else:
                shard_size *= 4
            total_allocated += shard_size
    return total_allocated


@task(mfu=list[float])
def parse_mfus(folder: str | Path, logfile: str, dry_run=False):
    if dry_run:
        return [100, 100, 100, 100]

    log_path = Path(folder) / logfile
    text = get_file_text(log_path)
    if text is None:
        return []
    flops = re.compile(r"TFLOP.s.device: (\d+[.]\d+)").findall(text)
    return list(map(float, flops))


@task(flops=float)
def parse_flops(folder: str | Path, logfile: str, dry_run=False):
    if dry_run:
        return 1000

    log_path = Path(folder) / logfile
    text = get_file_text(log_path)
    if text is None:
        return []
    flops = re.compile(r"Total TFLOPs: (\d+[.]\d+)").findall(text)
    assert len(flops) > 0, "Total TFLOPs per device not found"

    return list(map(float, flops))[0]
