#! /usr/bin/env bash

export UCX_TLS=^mm
export OMPI_MCA_plm=isolated
export CUDNN_HOME=/usr/lib/x86_64-linux-gnu
export PJRT_NPROC=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1

python run.py \
  --pp=2 \
  --tp=4 \
  --gpus=0 \
  --cpus=8 \
  --fbmem=0 \
  --sysmem=40 \
  --num-layers=4 \
  --interleave=2 \
  --nodes=1 \
  --num-steps=3 \
  --model-dims=128 \
  --sequence-length=128 \
  --num-heads=32 \
  --backend legate \
  --no-te \
  --profile \
  --dump dump \
  --dump-mpmd-passes \
  --sequence-parallel \
  --autoshard \
  --batch-size=8 \
  --microbatch-size=2 \
  --debug info
