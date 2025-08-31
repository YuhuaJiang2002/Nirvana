#!/bin/bash
echo $HOME
conda activate nirvana
conda env list

export HF_ENDPOINT="https://hf-mirror.com"
export TRITON_CACHE_DIR="/tmp/triton"

SUFFIX='29'
OUTPUT_DIR="./nirvana-1_3B-${SUFFIX}"

PY_ARGS=${PY_ARGS:-""}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export PYTHONPATH="$(pwd):$(pwd)/../"
echo "==== START TORCHRUN ===="

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun \
  --nproc-per-node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  pretrain-concat-nirvana.py \
  --train-cfg ./nirvana-finewebedu-1_3B.py \
  --llm ./nirvana_1_3B.json \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 64 \
  --tb-interval 128 \
  --gc-interval 2000 \
  --seed 1024 \
  --checkpoint-interval 1910 \
  --log-wandb \
  --wandb-project-name nirvana-1_3b-myds \
  --wandb-name nirvana-1_3b-myds-${SUFFIX} \
  --hf-interval 3820 \
  --max-keep-ckpts 1000 \
  --selective-recompute 0.0 \
  --dtype 'bf16' \
  --debug \
  --resume \
  ${PY_ARGS} \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "==== END TORCHRUN ===="
