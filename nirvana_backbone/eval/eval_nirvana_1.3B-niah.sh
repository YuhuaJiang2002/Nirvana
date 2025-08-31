#!/bin/bash
set -x

conda activate nirvana

export HF_ENDPOINT="https://hf-mirror.com"

# suffix='t3-align'
# suffix='nospecialmask'
suffix='h-rotate-21'

model_path="./nirvana-1_3B-${suffix}/hf-95500"
tokenizer_path="/cpfs02/shared/llmit6/liudawei/models/Llama-2-7b-chat-hf"
cp ${tokenizer_path}/tokenizer_config.json "$model_path"
cp ${tokenizer_path}/tokenizer.json "$model_path"
cp ${tokenizer_path}/tokenizer.model "$model_path"
cp -a ./configs/nirvana_1.3B-${suffix}/. "$model_path"
cp -a ./configs/infer_modeling/. "$model_path"

if [ -z "$RANK" ]; then
    RANK=0
fi

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

if [ "$RANK" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=12345 --multi_gpu --num_processes 8 -m evals.harness --model hf \
        --model_args pretrained=${model_path},dtype=bfloat16,trust_remote_code=True,max_length=32768 \
        --tasks niah_single_1,niah_single_2,niah_single_3 \
        --metadata='{"max_seq_lengths":[1024,2048,4096,8192]}' \
        --batch_size 2 \
        --num_fewshot 0 \
        --device cuda \
        --show_config \
        --trust_remote_code \
        --output_path ./evaluation_results/nirvana-1.3B-${suffix}-niah-single.json
else
    echo "skip evaluation"
fi

# --metadata='{"max_seq_lengths":[1024,2048,4096,8192]}' \
# piqa,social_iqa
#  hellaswag,winogrande,piqa,boolq,copa,lambada_openai,wikitext,arc_easy,arc_challenge,sciq,openbookqa
# python -m evals.harness --model hf \
#     --model_args pretrained=${model_path},dtype=bfloat16 \
#     --tasks hellaswag,winogrande,piqa,boolq,copa,lambada_openai,wikitext,arc_easy,arc_challenge,sciq,openbookqa \
#     --batch_size 64 \
#     --num_fewshot 0 \
#     --device cuda \
#     --show_config

# --trust_remote_code \
# python -m evals.harness --model hf \
#     --model_args pretrained=${model_path},dtype=bfloat16 \
#     --tasks  \
#     --batch_size 64 \
#     --num_fewshot 0 \
#     --device cuda \
#     --show_config

# accelerate launch --multi_gpu --num_processes 2 \
#     -m lm_eval --model hf \
#     --tasks lambada_openai,arc_easy \
#     --model_args 
# parallelize=True \
#     --batch_size 16
# python -m evals.harness --model hf \
#     --model_args pretrained=${model_path},dtype=bfloat16 \
#     --tasks hellaswag \
#     --batch_size 64 \
#     --num_fewshot 0 \
#     --device cuda \
#     --show_config

# accelerate launch \
#     -m evals.harness --model hf \
#     --model_args pretrained=${model_path},dtype=bfloat16 \
#     --tasks lambada_openai,arc_easy \
#     --model_args parallelize=True \
#     --batch_size 16
