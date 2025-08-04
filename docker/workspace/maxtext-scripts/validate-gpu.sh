#! /usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export PJRT_NPROC=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1

num_gpus_available=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
gpus=${1:-$num_gpus_available}

python ./run.py \
   --cpus 2 \
   --gpus $gpus \
   --fbmem 20 \
   --sysmem 20 \
   --nodes 1 \
   --dp 1 \
   --tp $gpus \
   --pp 1 \
   --ep 1 \
   --fsdp 1 \
   --num-layers 4 \
   --batch-size 4 \
   --microbatch-size 2 \
   --sequence-length 256 \
   --num-steps 20 \
   --use-iota-embed \
   --xla-rs-threshold=51200 \
   --attention cudnn_flash_te \
   --remat minimal \
   --replicate-small-params \
   --use-nccl-comm-split \
   --dump dump \
   --dump-mpmd-passes \
   --autoshard \
   --backend multimesh \
   --debug info
