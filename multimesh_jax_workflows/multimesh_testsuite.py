#! /usr/bin/env python

from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from eos_workflows import Experiment, Testsuite, srun, task
from eos_workflows.images import save_sqsh_image

from .maxtext import get_maxtext_results, parse_maxtext_results, run_maxtext


@task(folder=str)
def make_job_folder(row, name: str, base_folder: Optional[str] = None):
    folder = Path(base_folder) / name / Experiment.folder(row)
    return str(folder.resolve())


@task(jobid=str, rc=int)
def submit_jobs(row, dry_run: bool = False, **kwargs):
    if "remote" in row:
        fxn = partial(run_maxtext.remote, remote=row.remote)
    else:
        fxn = run_maxtext

    output, jobid, rc = fxn(
        folder=row.job_folder,
        dry_run=dry_run,
        **run_maxtext.filter_params(row),
    )
    return jobid, rc


@task(timings=object, losses=object, memory=int, flops=float, rc=int)
def parse_results(row, dry_run: bool = False, **kwargs):
    if "remote" in row:
        fxn = partial(parse_maxtext_results.remote, remote=row.remote)
    else:
        fxn = parse_maxtext_results
    timings, losses, memory, flops = fxn(job_folder=row.job_folder, dry_run=dry_run)
    rc = 0
    return timings, losses, memory, flops, rc


@task(timings=object, losses=object, memory=int, flops=float, rc=int)
def run_and_parse_results(row, dry_run: bool = False, **kwargs):
    if "remote" in row:
        fxn = partial(get_maxtext_results.remote, remote=row.remote)
    else:
        fxn = get_maxtext_results

    timings, losses, memory, flops, _, rc = fxn(
        **get_maxtext_results.filter_options(row),
        job_folder=row.job_folder,
        image=row.image,
        dry_run=dry_run,
    )
    return timings, losses, memory, flops, rc


@task(results=object)
def run_multimesh_testsuite(
    name: str,
    image: str,
    tag: Optional[str] = None,
    dry_run: bool = False,
    launcher: str = "sbatch",
    run_jobs: bool = True,
    wait_on_jobs: bool = True,
    use_remote: bool = True,
    model: Optional[str] = None,
    perf_only: bool = False,
    save_image: bool = True,
):
    expected = Experiment.create_df(
        dict(
            model="gpt3-24layers",
            dp=1,
            gpus=16,
            batch_size=128,
            microbatch_size=4,
            pp=2,
            tp=8,
            ep=1,
            hoist_loop_convert=True,
            remote="eos",
            mean_time=9.3,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=59e9,
            tags=["small"],
        ),
        dict(
            model="gpt3-24layers",
            gpus=32,
            batch_size=256,
            microbatch_size=4,
            dp=2,
            pp=2,
            tp=8,
            ep=1,
            hoist_loop_convert=True,
            remote="eos",
            mean_time=9.4,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=59e9,
            tags=["small"],
        ),
        dict(
            model="gpt3-24layers",
            gpus=32,
            batch_size=256,
            microbatch_size=4,
            dp=2,
            pp=2,
            tp=8,
            ep=1,
            schedule="zero-bubble-h2",
            hoist_loop_convert=True,
            remote="eos",
            mean_time=10.0,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=71e9,
            tags=["small", "zero-bubble"],
        ),
        dict(
            model="gpt3-175b",
            gpus=64,
            batch_size=128,
            microbatch_size=4,
            dp=1,
            pp=8,
            tp=8,
            ep=1,
            hoist_loop_convert=False,
            remote="eos",
            mean_time=9.65,
            stdev_time=0.05,
            mean_tolerance=0.1,
            stdev_tolerance=0.1,
            memory=73e9,
            tags=["large"],
        ),
        dict(
            model="gpt3-175b",
            gpus=128,
            batch_size=256,
            microbatch_size=4,
            dp=2,
            pp=8,
            tp=8,
            ep=1,
            hoist_loop_convert=False,
            remote="eos",
            mean_time=9.85,
            stdev_time=0.05,
            mean_tolerance=0.1,
            stdev_tolerance=0.1,
            memory=75e9,
            tags=["large"],
        ),
        dict(
            model="llama3-70b",
            gpus=64,
            batch_size=128,
            microbatch_size=2,
            dp=1,
            pp=16,
            tp=4,
            ep=1,
            hoist_loop_convert=True,
            remote="eos",
            mean_time=8.9,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=72e9,
            tags=["large"],
        ),
        dict(
            model="llama3-70b",
            gpus=64,
            batch_size=64,
            microbatch_size=1,
            dp=1,
            pp=16,
            tp=4,
            ep=1,
            schedule="zero-bubble-h2",
            hoist_loop_convert=True,
            remote="eos",
            mean_time=4.7,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=70e9,
            tags=["large", "zero-bubble"],
        ),
        dict(
            model="llama3-8b",
            gpus=16,
            batch_size=32,
            microbatch_size=2,
            num_steps=12 if perf_only else 750,
            eval_interval=3 if perf_only else 50,
            eval_steps=2 if perf_only else 25,
            time=45,
            dataset="synthetic" if perf_only else "c4",
            dp=2,
            pp=4,
            tp=2,
            ep=1,
            hoist_loop_convert=True,
            remote="eos",
            mean_time=0.98,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            tags=["small", "convergence"],
        ),
        dict(
            model="mixtral-8x7b",
            gpus=16,
            fbmem=72,
            batch_size=32,
            microbatch_size=8,
            num_steps=12,
            dp=1,
            pp=2,
            tp=1,
            ep=8,
            capacity_factor=1,
            hoist_loop_convert=True,
            remote="eos",
            mean_time=2.10,
            stdev_time=0.05,
            mean_tolerance=0.05,
            stdev_tolerance=0.05,
            memory=58e9,
            tags=["moe", "small"],
        ),
        tag=tag,
    )

    DEFAULT_CAPACITY_FACTOR = -1
    DEFAULT_TIME = 0
    DEFAULT_FBMEM = 77
    DEFAULT_NUM_STEPS = 12
    DEFAULT_EVAL_INTERVAL = -1
    DEFAULT_EVAL_STEPS = -1
    DEFAULT_DATASET = "synthetic"
    DEFAULT_SCHEDULE = "prefetch-wavefront"

    expected["capacity_factor"] = expected["capacity_factor"].fillna(DEFAULT_CAPACITY_FACTOR).astype(float)
    expected["time"] = expected["time"].fillna(DEFAULT_TIME).astype(int)
    expected["fbmem"] = expected["fbmem"].fillna(DEFAULT_FBMEM).astype(int)
    expected["num_steps"] = expected["num_steps"].fillna(DEFAULT_NUM_STEPS).astype(int)
    expected["eval_interval"] = expected["eval_interval"].fillna(DEFAULT_EVAL_INTERVAL).astype(int)
    expected["eval_steps"] = expected["eval_steps"].fillna(DEFAULT_EVAL_STEPS).astype(int)
    expected["dataset"] = expected["dataset"].fillna(DEFAULT_DATASET)
    expected["schedule"] = expected["schedule"].fillna(DEFAULT_SCHEDULE)
    expected["launcher"] = launcher

    if model is not None:
        expected = expected[expected["model"] == model]

    if save_image:
        remotes = expected.remote.unique()
        sqshs = {}
        for remote in remotes:
            sqshs[remote] = save_sqsh_image.remote(
                image=image, remote=remote, dry_run=dry_run
            )
        expected["image"] = expected["remote"].apply(lambda x: sqshs[x])
    else:
        expected["image"] = image

    folder_attrs = [
        "dp",
        "tp",
        "ep",
        "pp",
        "schedule",
        "model",
        "num_steps",
        "dataset",
        "hoist_loop_convert",
        "gpus",
        "batch_size",
        "microbatch_size",
        "time",
        "remote",
    ]
    expected["job_folder"] = expected[folder_attrs].apply(
        lambda row: make_job_folder.remote(row=row, remote=row.remote, name=name),
        axis=1,
    )

    if not use_remote:
        expected.drop("remote", axis=1, inplace=True)

    fields = [
        "mean_time",
        "stdev_time",
        "memory",
        "mean_tolerance",
        "stdev_tolerance",
    ]

    e = Experiment.create(name=name, df=expected.drop(columns=fields))
    if launcher == "srun":
        to_apply = run_and_parse_results.partial(job_name=name, dry_run=dry_run)
        result = e.apply(to_apply, name="run_and_parse_results").data
    else:
        jobs_csv = f"{name}.jobs.csv"
        if run_jobs:
            to_apply = submit_jobs.partial(job_name=name, dry_run=dry_run)
            jobs = e.apply(to_apply, name="submit_jobs")
            jobs.data.to_csv(jobs_csv)
        else:
            jobs_data = pd.read_csv(jobs_csv)
            jobs = Experiment.create(name=name, df=jobs_data)

        if wait_on_jobs and "remote" in jobs.data.columns:
            unique_remotes = (
                jobs.data.groupby("remote").agg(dict(jobid=list)).reset_index()
            )
            for index, row in unique_remotes.iterrows():
                srun.remote(
                    application="wait",
                    time=1,
                    remote=row.remote,
                    dependencies=row.jobid,
                    cmd=["echo", "hello"],
                    dry_run=dry_run,
                )

        to_apply = parse_results.partial(dry_run=dry_run)
        result = jobs.apply(to_apply, name="parse_results").data

    # slice away the first few
    result["mean_time"] = result.apply(lambda x: np.mean(x.timings[3:]), axis=1)
    result["stdev_time"] = result.apply(lambda x: np.std(x.timings[3:]), axis=1)
    # todo, fix this for fp8
    result["mfu"] = result.apply(lambda x: x.flops / x.mean_time, axis=1) / 989

    def validate_mean(expected, test, row):
        return (test - expected) < row.mean_tolerance

    def validate_stdev(expected, test, row):
        return (test - expected) < row.stdev_tolerance

    def validate_memory(expected, test, row):
        return test < expected

    def validate_timings(test, row):
        return len(test) == row.num_steps

    def validate_losses(test, row):
        # TODO: implement better loss checker
        if len(test) != row.num_steps:
            return False
        elif row.dataset != DEFAULT_DATASET and row.num_steps >= 100:
            return test[100] < test[25]
        return True

    def validate_rc(test, row):
        return test == 0

    test = (
        Testsuite(result, attrs=list(e.data.columns))
        .compare(
            expected,
            mean_time=validate_mean,
            stdev_time=validate_stdev,
            memory=validate_memory,
        )
        .validate(rc=validate_rc, losses=validate_losses)
        .finalize()
    )

    perf_dashboard = result[
        [
            "model",
            "pp",
            "tp",
            "dp",
            "schedule",
            "gpus",
            "mfu",
            "flops",
            "mean_time",
            "microbatch_size",
            "memory",
            "dataset",
            "remote",
            "job_folder",
        ]
    ]
    perf_dashboard.to_csv(f"{name}.perf_dashboard.csv")

    test.data.to_csv(f"{name}.test.csv")
    print(test.data)
    print(test.summary)
    print(test.num_failures)
    return test
