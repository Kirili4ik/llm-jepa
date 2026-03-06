#!/bin/bash
set -e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)


torchrun --nproc_per_node=$NGPUS --master_port=29508 finetune.py \
  --train_file datasets/synth_train.jsonl \
  --output_dir=./fine-tuned-llama1b-regular-synth \
  --num_epochs=4 --finetune_seed=82 --regular \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=2e-5 \
  --batch_size=8 --grad_accum=8

CUDA_VISIBLE_DEVICES=$FIRST_GPU python evaluate.py \
  --model_name=$(pwd)/fine-tuned-llama1b-regular-synth \
  --input_file=datasets/synth_test.jsonl \
  --output_file=eval_llama1b_regular_synth.jsonl \
  --original_model_name=meta-llama/Llama-3.2-1B-Instruct \
  --nosplit_data --split_tune_untune --device_map=cuda:0 --max_new_tokens=128 | tee -a output.txt
