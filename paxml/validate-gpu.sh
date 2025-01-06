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

export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_RANK}

python run.py \
  --dp=1 \
  --fsdp=1 \
  --pp=1 \
  --tp=2 \
  --gpus=1 \
  --cpus=1 \
  --nodes=2 \
  --fbmem=5 \
  --sysmem=40 \
  --num-layers=4 \
  --batch-size=8 \
  --microbatch-size=2 \
  --num-steps=3 \
  --model-dims=512 \
  --num-heads=32 \
  --backend=legate \
  --te \
  --profile \
  --dump=dump \
  --dump-mpmd-passes \
  --only-fuse-loop-tasks \
  --sequence-parallel \
  --autoshard \
  --debug info
