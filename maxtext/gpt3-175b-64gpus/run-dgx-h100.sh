#!/bin/bash

export PJRT_NPROC=16
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1


RANK=${OMPI_COMM_WORLD_RANK}
export CUDA_VISIBLE_DEVICES=${RANK}

to_profile=$(( RANK % 8))
if [ "$to_profile" == "0" ]; then
  nsys_cmd="nsys profile --trace=nvtx -o nsys_${RANK} --force-overwrite=true"
else
  nsys_cmd=""
fi

cpus=(0-13,112-125  14-27,126-139  28-41,140-153  42-55,154-167  56-69,168-181  70-83,182-195  84-97,196-209  98-111,210-223)
mems=(0 0 0 0 1 1 1 1)

intra_node_rank=$(( RANK % 8 ))
cpu_binding="${cpus[$intra_node_rank]}"
mem_binding="${mems[$intra_node_rank]}"

export MULTIMESH_HOIST_CONVERT=1

$nsys_cmd \
numactl --physcpubind $cpu_binding \
	--membind $mem_binding \
python `pwd`/run.py \
    --cpus 1 \
    --gpus 1 \
    --fbmem 77 \
    --nodes 64 \
    --pp 8 \
    --tp 8 \
    --dp 1 \
    --fsdp 1 \
    --num-layers 96 \
    --interleave 12 \
    --model-name gpt3-175b \
    --model-dims 12288 \
    --num-heads 96 \
    --xla-rs-threshold=51200 \
    --attention=cudnn_flash_te \
    --remat minimal \
    --sequence-length=2048 \
    --batch-size 128 \
    --microbatch-size 4 \
    --network ucx \
    --profile \
    --replicate-small-params \
    --schedule wavefront \
    --load-balance-embeddings \
    --no-sequence-parallel \
    --autoshard \
    --num-steps 8 \
    --backend multimesh \
    --debug info >& ${RANK}.out
