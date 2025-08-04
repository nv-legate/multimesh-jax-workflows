#!/bin/bash

export PJRT_NPROC=16
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=1
export TF_CPP_VMODULE=""
export REALM_FREEZE_ON_ERROR=0

export ENABLE_TE=1
export ENABLE_TE_SP=1
export NVTE_FUSED_ATTN=1

RANK=$SLURM_PROCID

if [ "$RANK" == "0" ]; then
  dump="{dump_flags}"
fi

if [ "$RANK" == "1" ]; then
  debug=debug
else
  debug=info
fi

cpus=(0-13,112-125  14-27,126-139  28-41,140-153  42-55,154-167  56-69,168-181  70-83,182-195  84-97,196-209  98-111,210-223)
mems=(0 0 0 0 1 1 1 1)

intra_node_rank=$(( RANK % 8 ))
cpu_binding="${{cpus[$intra_node_rank]}}"
mem_binding="${{mems[$intra_node_rank]}}"

export CUDA_VISIBLE_DEVICES=$intra_node_rank

echo $cpu_binding
echo $mem_binding

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_graph_level=0
                --xla_gpu_all_reduce_combine_threshold_bytes=1073741824
                --xla_gpu_all_gather_combine_threshold_bytes=1073741824
                --xla_gpu_reduce_scatter_combine_threshold_bytes=51200
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization,hlo-verifier"

cd /workspace/cwd

mkdir $RANK
cd $RANK

run() {{
numactl --physcpubind $cpu_binding \
	--membind $mem_binding \
python /opt/maxtext/run.py \
    --cpus 1 \
    --gpus 1 \
    --fbmem {fbmem} \
    --num-layers {num_layers} \
    --model-name {model} \
    --xla-rs-threshold=51200 \
    --attention=cudnn_flash_te \
    --sequence-length={sequence_length} \
    --capacity-factor={capacity_factor} \
    --batch-size {batch_size} \
    --microbatch-size {microbatch_size} \
    --interleave {interleave} \
    --num-steps {num_steps} \
    --{use_tfds_dataset}use-tfds-dataset \
    --dataset-name {dataset_name} \
    --dataset-path {dataset_path} \
    --tokenizer-path {tokenizer_path} \
    --{perform_eval}perform-eval \
    --eval-interval {eval_interval} \
    --eval-steps {eval_steps} \
    --eval-batch-size {eval_batch_size} \
    --use-nccl-comm-split \
    --only-fuse-loop-tasks \
    --network ucx \
    --remat minimal \
    --schedule {schedule} \
    --{hoist_loop_convert}hoist-loop-convert \
    --autoshard \
    $dump \
    --debug $debug \
    --backend multimesh \
    -logfile jax_%.log \
    --nodes {num_gpus} \
    --dp {dp} \
    --pp {pp} \
    --tp {tp} \
    --ep {ep} \
    --fsdp 1 \
    $to_output
}}


if [ "$RANK" == "0" ]; then
  run | tee $RANK.out
else
  run >& $RANK.out
fi
