#! /usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export PJRT_NPROC=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1

python ./run.py \
   --cpus 8 \
   --gpus 0 \
   --fbmem 0 \
   --sysmem 5 \
   --nodes 1 \
   --dp 2 \
   --tp 2 \
   --ep 1 \
   --pp 2 \
   --fsdp 1 \
   --num-layers 2 \
   --interleave 1 \
   --batch-size 8 \
   --microbatch-size 2 \
   --sequence-length 128 \
   --attention dot_product \
   --model-dims 256 \
   --num-steps 3 \
   --use-iota-embed \
   --model-name mixtral-small \
   --xla-rs-threshold=51200 \
   --remat minimal \
   --replicate-small-params \
   --use-nccl-comm-split \
   --load-balance-decoder-norm \
   --dump dump \
   --autoshard \
   --backend multimesh \
   --debug info
