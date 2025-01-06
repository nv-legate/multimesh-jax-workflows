#! /usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PJRT_NPROC=16
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"
export LEGATE_XLA_HOIST_CONVERT=1


RANK=${SLURM_PROCID}
CUDA_VISIBLE_DEVICES=$(( RANK % 8 ))
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
cpus=(0-13,112-125  14-27,126-139  28-41,140-153  42-55,154-167  56-69,168-181  70-83,182-195  84-97,196-209  98-111,210-223)
mems=(0 0 0 0 1 1 1 1)

intra_node_rank=$(( RANK % 8 ))
cpu_binding="${cpus[$intra_node_rank]}"
mem_binding="${mems[$intra_node_rank]}"

mkdir -p logs/checkpoints

if [[ $intra_node_rank -eq 0 ]]; then
  nsys_cmd="nsys profile --trace=nvtx -o nsys-${RANK} --sample=cpu --force-overwrite=true"
else
  nsys_cmd=""
fi

echo "cmd=$nsys_cmd"

$nsys_cmd \
numactl --physcpubind $cpu_binding \
        --membind $mem_binding \
python ./run.py \
   --cpus 1 \
   --gpus 1 \
   --nodes 64 \
   --dp 1 \
   --fsdp 1 \
   --pp 8 \
   --tp 8 \
   --fbmem 75 \
   --zcmem 50 \
   --sysmem 4 \
   --use-nccl-comm-split \
   --num-layers 96 \
   --num-heads 96 \
   --interleave 12 \
   --network ucx \
   --model-dims 12288 \
   --remat save_dot_only \
   --batch-size 128 \
   --paxml-config paxml.tasks.lm.params.nvidia.NVIDIA1_3B \
   --sequence-parallel \
   --load-balance-embeddings \
   --optimizer adafactor \
   --only-fuse-loop-tasks \
   --schedule prefetch-wavefront \
   --cache-parallelism 3 \
   --autoshard \
   --profile \
   --microbatch-size 4 \
   --num-steps 8 \
   --backend legate \
   -logfile jax_%.log \
   --debug info >& test.out
