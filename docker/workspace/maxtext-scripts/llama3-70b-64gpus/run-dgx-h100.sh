#!/bin/bash

# launch file to run llama3-70b

# Note: this file is configured to be launched via slurm tasks
# Note: use your appropriate launcher environment var to determine the rank of the GPU
RANK=${SLURM_PROCID}

GPUS_PER_NODE=8

# xla by default allocates many threads, which can conflict with
# the threads used by the plugin client and makes backtraces
# harder to debug and interpret.
export PJRT_NPROC=16
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

# prefix launch command with nsys_cmd to collect nsight traces 
# nsys_cmd="nsys profile --trace=nvtx -o nsys_${RANK} --force-overwrite=true"

# limit the visibility of devices to only the one that the process will be running (1 process/device)
intra_node_rank=$(( RANK % GPUS_PER_NODE ))
export CUDA_VISIBLE_DEVICES=${intra_node_rank}

# configure cpu and memory region affinity for executing on DGX H100s
cpus=(0-13,112-125  14-27,126-139  28-41,140-153  42-55,154-167  56-69,168-181  70-83,182-195  84-97,196-209  98-111,210-223)
mems=(0 0 0 0 1 1 1 1)

cpu_binding="${cpus[$intra_node_rank]}"
mem_binding="${mems[$intra_node_rank]}"

# run training
numactl --physcpubind $cpu_binding \
        --membind $mem_binding \
python /opt/maxtext/run.py \
    --fbmem 77 \
    --cpus 1 \
    --gpus 1 \
    --nodes 64 \
    --dp 1 \
    --fsdp 1 \
    --tp 4 \
    --pp 16 \
    --interleave 5 \
    --model-name llama3-70b \
    --num-layers 80 \
    --sequence-length=4096 \
    --batch-size 128 \
    --microbatch-size 2 \
    --remat minimal \
    --attention=cudnn_flash_te \
    --xla-rs-threshold=51200 \
    --use-nccl-comm-split \
    --replicate-small-params \
    --hoist-loop-convert \
    --network ucx \
    --schedule prefetch-wavefront \
    --autoshard \
    --num-steps 10 \
    --debug info \
    --backend multimesh \
    -logfile jax_%.log \
    >& ${RANK}.out
