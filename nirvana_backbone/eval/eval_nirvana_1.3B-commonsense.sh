#!/bin/bash
set -x

conda activate nirvana

export HF_ENDPOINT="https://hf-mirror.com"

suffix=29

model_path="./nirvana-1_3B/hf-95500"

tokenizer_path="<YOUR_PATH>"
cp ${tokenizer_path}/tokenizer_config.json "$model_path"
cp ${tokenizer_path}/tokenizer.json "$model_path"
cp ${tokenizer_path}/tokenizer.model "$model_path"
cp -a ./configs/nirvana_1.3B-${suffix}/. "$model_path"
cp -a ./configs/infer_modeling/. "$model_path"

if [ -z "$RANK" ]; then
    RANK=0
fi

if [ "$RANK" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=12344 --multi_gpu --num_processes 8 -m evals.harness --model hf \
        --model_args pretrained=${model_path},dtype=bfloat16,trust_remote_code=True \
        --tasks social_iqa,piqa,wikitext,lambada_openai,hellaswag,winogrande,boolq,arc_easy,arc_challenge \
        --batch_size 32 \
        --num_fewshot 0 \
        --device cuda \
        --show_config \
        --output_path ./evaluation_results/nirvana-1.3B-${suffix}-arc.json
else
    echo "skip evaluation"
fi

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
